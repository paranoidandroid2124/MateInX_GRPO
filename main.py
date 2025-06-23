import torch
from pathlib import Path
import yaml
from typing import Any, Tuple, Dict, List, Optional
import os
from dotenv import load_dotenv
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error

from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from trl import GRPOConfig

from reward.chess_reward import ChessReward
from trainers.custom_grpo_trainer import CustomChessGRPOTrainer
from unsloth import FastLanguageModel
from utils.prompting import chess_prompt_formatter
from utils.context import simple_context_updater
from data.data import load_dataset_from_json


load_dotenv()
set_verbosity_error()

def validate_config(cfg: Dict[str, Any], section: str) -> bool:
    """설정 파일의 값들을 검증합니다."""
    required_keys = {
        "training": [
            "model_name", "learning_rate", "num_train_epochs", 
            "rollout_batch_size", "num_generations", "data_path"
        ],
        "reward": [
            "format_pass_base_reward", "move_correct_base_reward", 
            "move_incorrect_base_penalty"
        ]
    }
    
    # 필수 키 체크
    if section in required_keys:
        for key in required_keys[section]:
            if key not in cfg:
                raise KeyError(f"Missing required config key: {section}.{key}")
    
    # 값 범위 검증
    if section == "training":
        if cfg.get("learning_rate", 0) <= 0:
            raise ValueError("learning_rate must be positive")
        if cfg.get("num_generations", 0) <= 0:
            raise ValueError("num_generations must be positive")
        if cfg.get("rollout_batch_size", 0) <= 0:
            raise ValueError("rollout_batch_size must be positive")
            
    return True

with open("config/default.yaml") as f:
    CFG = yaml.safe_load(f)
    TRAINING_CFG = CFG["training"]
    REWARD_CFG = CFG["reward"]
    
    # 설정 검증
    validate_config(TRAINING_CFG, "training")
    validate_config(REWARD_CFG, "reward")
    print("✅ Configuration validation passed")


def load_tokenizer_and_model() -> Tuple[Any, Any]:
    # YAML 설정 파일에서 모델 설정 가져오기
    model_config = {
        "model_name": TRAINING_CFG.get("model_name", "unsloth/Qwen3-1.7B-Base"),
        "max_seq_length": TRAINING_CFG.get("max_seq_length", 2048),
        "load_in_4bit": TRAINING_CFG.get("load_in_4bit", False),
        "dtype": TRAINING_CFG.get("dtype", "bfloat16"),
        "gradient_checkpointing": TRAINING_CFG.get("gradient_checkpointing", True),
    }

    # YAML 설정 파일에서 훈련 설정 가져오기
    training_config = {
        "use_lora": TRAINING_CFG.get("use_lora", True),
        "lora_params": TRAINING_CFG.get("lora_params", {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        }),
        "seed": TRAINING_CFG.get("seed", 42),
    }

    print(f"Using model: {model_config['model_name']}")

    model_name_or_path = model_config.get("model_name")
    max_seq_length = model_config.get("max_seq_length", 2048)
    load_in_4bit = model_config.get("load_in_4bit", False)
    dtype_str = model_config.get("dtype", None)
    
    # getattr으로 torch 모듈에서 dtype 가져오기 (안전하게)
    dtype = None
    if dtype_str and isinstance(dtype_str, str):
        if hasattr(torch, dtype_str):
            dtype = getattr(torch, dtype_str)
        else:
            print(f"Warning: dtype '{dtype_str}' not found in torch. Using default.")
    elif dtype_str is not None: # 문자열이 아닌 경우 (잘못된 설정)
        print(f"Warning: Invalid dtype configuration '{dtype_str}'. Using default.")


    # 기본 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=0.8,
        trust_remote_code=True, # 모델에 따라 필요
    )

    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
                print("⚠️ Using UNK token as pad token")
            else:
                raise ValueError("Neither EOS nor UNK token is available to set as pad_token. Please configure tokenizer manually.")

    if tokenizer.pad_token_id is not None:
         model.config.pad_token_id = tokenizer.pad_token_id
    elif tokenizer.pad_token:
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        raise ValueError("pad_token and pad_token_id are both None. Tokenizer is not configured for padding.")

    # 현재 프롬프트 구조에 최적화된 Qwen3 chat template
    # 시스템/퓨샷/멀티턴 모두 지원하면서 간단하고 효율적
    chat_template = """{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"""
    
    tokenizer.chat_template = chat_template
    print("✅ Chat template configured")
    
    generation_config = getattr(model, "generation_config", None)
    if generation_config:
        # 체스 추론을 위한 최적의 생성 설정 적용
        generation_config.temperature = 0.7
        generation_config.top_p = 0.8
        generation_config.top_k = 20
        generation_config.min_p = 0.0
        
        # thinking 관련 토큰들 억제 (토큰 절약)
        think_start_token_id = tokenizer.convert_tokens_to_ids("<think>")
        think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
        
        # bad_words_ids로 thinking 토큰 억제
        bad_words_ids = []
        if think_start_token_id != tokenizer.unk_token_id:
            bad_words_ids.append([think_start_token_id])
        if think_end_token_id != tokenizer.unk_token_id:
            bad_words_ids.append([think_end_token_id])
            
        if bad_words_ids:
            generation_config.bad_words_ids = bad_words_ids
        
        print("✅ Generation config applied")
    else:
        print("⚠️ Generation config not found")

    use_lora = training_config.get("use_lora", True)
    if not use_lora:
        print("LoRA is not used. Returning base model and tokenizer.")
        return tokenizer, model

    #LoRA 파라미터 가져오기
    lora_params = training_config.get("lora_params", {})
    
    print(f"Applying LoRA with parameters: r={lora_params.get('r', 16)}, alpha={lora_params.get('lora_alpha', 32)}")

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_params.get("r", 16)),
        lora_alpha=int(lora_params.get("lora_alpha", 32)),
        target_modules=lora_params.get("target_modules", None),
        lora_dropout=float(lora_params.get("lora_dropout", 0.05)),
        bias=lora_params.get("bias", "none"),
        use_gradient_checkpointing=model_config.get("gradient_checkpointing", True),
        random_state=training_config.get("seed", 42),
        max_seq_length=max_seq_length,
    )
    
    print("LoRA adapters injected via Unsloth.")
    if hasattr(peft_model, 'print_trainable_parameters'):
        peft_model.print_trainable_parameters()

    return tokenizer, peft_model


def run_training(cfg: Dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("Set GPU memory usage limit to 90%.")
    accel = Accelerator()
    
    # 메모리 최적화 설정 가져오기
    memory_config = cfg.get("memory_optimization", {})
    clear_cache_every_n_steps = memory_config.get("clear_cache_every_n_steps", 5)
    force_gc_every_n_steps = memory_config.get("force_gc_every_n_steps", 10)
    max_concurrent_trajectories = memory_config.get("max_concurrent_trajectories", 20)
    
    print(f"메모리 최적화 설정:")
    print(f"  - 캐시 정리 주기: {clear_cache_every_n_steps} 스텝")
    print(f"  - GC 강제 실행 주기: {force_gc_every_n_steps} 스텝")
    print(f"  - 최대 동시 궤적 수: {'제한 없음' if max_concurrent_trajectories == -1 else max_concurrent_trajectories}")
    
    grpo_cfg = GRPOConfig(
        output_dir=Path(cfg["output_dir"]).as_posix(),
        learning_rate=float(cfg["learning_rate"]),
        num_generations=cfg["num_generations"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        per_device_train_batch_size=cfg["rollout_batch_size"],
        num_train_epochs=cfg["num_train_epochs"],
        beta=cfg["beta"],
        max_prompt_length=cfg["max_prompt_len"],
        max_completion_length=cfg["max_completion_len"],
        remove_unused_columns=False,
        logging_steps=1,
    )

    tokenizer, model = load_tokenizer_and_model()
    reward_fn = ChessReward(config=REWARD_CFG)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=grpo_cfg.learning_rate,
    )

    dataset = load_dataset_from_json(TRAINING_CFG["data_path"])
    trainer = CustomChessGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_cfg,
        puzzle_reward_model=reward_fn,
        puzzle_dataset_for_rollout=dataset,
        prompt_formatter=chess_prompt_formatter,
        context_updater=simple_context_updater,
        optimizer=opt,
        lr_scheduler=None,
        training_config=cfg,  # TRAINING_CFG 전체를 전달
    )
    
    puzzle_batch_size = CFG["puzzle_batch_size"]
    
    model.train()
    total_batches = len(dataset) // puzzle_batch_size

    for epoch in range(int(grpo_cfg.num_train_epochs)):
        print(f"\n=== Epoch {epoch + 1}/{grpo_cfg.num_train_epochs} ===")
        
        # 에포크 시작 시 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # GPU 동기화

        for idx in range(
            0,
            len(dataset),
            puzzle_batch_size * accel.num_processes,
        ):
            start = idx + puzzle_batch_size * accel.process_index
            end = start + puzzle_batch_size
            batch = dataset[start:end]
            if not batch:
                continue
            global_step = idx // (
                puzzle_batch_size * accel.num_processes
            ) + 1

            # 메모리 사용량 모니터링
            gpu_mem_mb = (
                torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            )
            gpu_max_mb = (
                torch.cuda.get_device_properties(0).total_memory / 1e6 if torch.cuda.is_available() else 0
            )
            memory_usage_pct = (gpu_mem_mb / gpu_max_mb * 100) if gpu_max_mb > 0 else 0
            
            print(
                f"Epoch {epoch + 1} | Batch {global_step}/{total_batches} | GPU: {memory_usage_pct:.1f}%"
            )
            
            # 메모리 사용량이 80% 넘으면 강제 정리
            if memory_usage_pct > 80:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc
                gc.collect()

            with accel.accumulate(model):
                trajs = trainer.generate_rollout_for_grpo(batch)

                if not trajs:
                    print(f"[WARN] Batch {global_step} produced no trajectories; skipping.")
                    continue
                    
                # 설정 파일의 max_concurrent_trajectories 적용 (-1이면 제한 없음)
                total_trajectories = len(trajs)
                if max_concurrent_trajectories > 0 and total_trajectories > max_concurrent_trajectories:
                    print(f"⚠️ Too many trajectories ({total_trajectories}), truncating to {max_concurrent_trajectories}")
                    trajs = trajs[:max_concurrent_trajectories]
                elif max_concurrent_trajectories == -1:
                    print(f"📊 Processing {total_trajectories} trajectories (no limit)")
                else:
                    print(f"📊 Processing {total_trajectories} trajectories")
                
                loss = trainer.step(trajs)
                if accel.is_main_process and loss is not None:
                    print(
                        f"Epoch {epoch + 1} | Batch {global_step} | loss {loss.item():.4f}"
                    )

            # 설정 파일의 캐시 정리 주기 적용
            if torch.cuda.is_available() and global_step % clear_cache_every_n_steps == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # 설정 파일의 GC 강제 실행 주기 적용
            if global_step % force_gc_every_n_steps == 0:
                import gc
                gc.collect()
                    
            # trajectory 정리
            del trajs, batch
            if 'loss' in locals():
                del loss

    print("\nTraining finished.")
    
    # 모델 저장 코드 추가 (에러 처리 포함)
    try:
        output_dir = Path(cfg["output_dir"]) / "final_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n모델 저장 중: {output_dir}")
        
        # 1. LoRA 어댑터만 저장 (가장 가벼운 옵션)
        try:
            lora_path = output_dir / "lora"
            model.save_pretrained_merged(
                str(lora_path), 
                tokenizer, 
                save_method="lora"
            )
            print(f"✅ LoRA 어댑터 저장 완료: {lora_path}")
        except Exception as e:
            print(f"❌ LoRA 어댑터 저장 실패: {e}")
        
        # 2. 16비트 병합 모델 저장 (더 큰 파일 크기지만 바로 사용 가능)
        try:
            merged_path = output_dir / "merged_16bit"
            model.save_pretrained_merged(
                str(merged_path), 
                tokenizer, 
                save_method="merged_16bit"
            )
            print(f"✅ 16비트 병합 모델 저장 완료: {merged_path}")
        except Exception as e:
            print(f"❌ 16비트 병합 모델 저장 실패: {e}")
            
    except Exception as e:
        print(f"❌ 모델 저장 중 오류 발생: {e}")
        
    print("\n모델 저장 프로세스 완료!")

if __name__ == "__main__":
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        raise ValueError("WANDB_API_KEY not found in .env file.")
    wandb.login(key=wandb_key)

    wandb.init(
        project=CFG["project"],
        name=CFG["name"],
        config={**TRAINING_CFG, **REWARD_CFG}
    )


    print("====== Run Training. ======\n\n")
    run_training(TRAINING_CFG)
