import torch
from pathlib import Path
import yaml
import wandb
from typing import Any, Tuple, Dict, List, Optional
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import chess
from tqdm import tqdm

from transformers.utils.logging import set_verbosity_error

from reward.chess_reward import ChessReward
from trainers.custom_grpo_trainer import CustomChessGRPOTrainer
from utils.prompting import chess_prompt_formatter
from utils.context import simple_context_updater
from data.data import load_dataset_from_json


load_dotenv()
set_verbosity_error()

with open("config/default.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
    TRAINING_CFG = CFG["training"]
    REWARD_CFG = CFG["reward"]


def load_tokenizer_and_model() -> Tuple[Any, Any]:
    # main.py와 동일한 함수 사용
    from main import load_tokenizer_and_model as _load_tokenizer_and_model
    return _load_tokenizer_and_model()

def setup_wandb(cfg: Dict[str, Any]) -> None:
    """Initializes Wandb run."""
    if cfg.get("wandb_log", True):
        wandb.init(
            project=CFG["project"],
            name=CFG["name"] + "_test",
            config=cfg,
        )

def run_testing(cfg: Dict[str, Any], test_data_path: str) -> None:
    # 테스트 시 생성할 trajectory 수
    k_value = cfg.get("test_num_trajectories_k", 3)
    print(f"Running test with k={k_value} trajectories per puzzle.")
    tokenizer, model = load_tokenizer_and_model()
    reward_fn = ChessReward(config=REWARD_CFG)
    
    # 평가 데이터셋 로드
    test_dataset = load_dataset_from_json(test_data_path)
    
    print(f"Loaded test dataset from {test_data_path} with {len(test_dataset)} puzzles.")

    # Wandb 설정
    setup_wandb(cfg)

    # 로그 파일 설정 (main.py의 로깅 방식 참고)
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"test_results_{timestamp}.jsonl")

    print(f"Testing and logging results to: {log_file_path}")
    
    model.eval()  # 평가 모드 설정
    all_results = []  # 모든 퍼즐 결과를 저장할 리스트
    with torch.no_grad():
        for puzzle_idx, puzzle_raw_item in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing puzzles"):
            puzzle_as_dict = {
                "initial_context": puzzle_raw_item["FEN"],
                "solution_moves": puzzle_raw_item["MovesSAN"],  # 정답 수순 (키 이름을 MovesSAN으로 가정)
                "total_llm_steps": (len(puzzle_raw_item["MovesSAN"]) + 1) // 2,
            }
            themes_str = puzzle_raw_item.get("Themes", "")  # 테마 정보 (옵션)

            # CustomChessGRPOTrainer의 _execute_batched_trajectories 메서드 직접 사용
            # main.py의 궤적 실행 로직을 재사용하여 평가
            trainer = CustomChessGRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=None,  # config는 rollout에 직접 사용되지 않으므로 None으로 전달
                puzzle_reward_model=reward_fn,
                puzzle_dataset_for_rollout=[],  # 빈 리스트 전달
                prompt_formatter=chess_prompt_formatter,
                context_updater=simple_context_updater,
                training_config=cfg, # 훈련 설정을 전달하여 생성 파라미터 일관성 유지
            )
            # k_value 만큼의 trajectory 생성
            _, _, traj_result_dicts, _ = trainer._execute_batched_trajectories(
                puzzle_as_dict, 
                themes_str, 
                num_trajectories=k_value
            )

            # 생성된 trajectory 결과들 중에서 가장 높은 점수를 받은 결과 선택
            if traj_result_dicts:
                # score 기준으로 가장 좋은 trajectory 선택
                best_traj_result = max(traj_result_dicts, key=lambda x: x.get("score", float('-inf')))
                
                response_ids = best_traj_result.get("ids")
                model_response_text = tokenizer.decode(response_ids, skip_special_tokens=True) \
                    if response_ids is not None and response_ids.numel() > 0 \
                    else "N/A"
                final_reward = best_traj_result.get("score", float('-inf'))
                
                all_results.append({
                    "puzzle_id": puzzle_idx,
                    "fen": puzzle_as_dict["initial_context"],
                    "themes": themes_str,
                    "reward": final_reward,
                    "model_response": model_response_text,
                })
            else:  # 오류
                all_results.append({
                    "puzzle_id": puzzle_idx,
                    "fen": puzzle_as_dict["initial_context"],
                    "themes": themes_str,
                    "reward": -1000.0,  # 매우 낮은 점수로 실패 표시
                    "error": "No trajectory generated",
                    "model_response": "N/A",
                })

    # 모든 결과 로깅
    with open(log_file_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    # Wandb 종료
    wandb.finish()

    print(f"Testing complete. Results logged to {log_file_path}")


if __name__ == "__main__":
    test_data_path = "data/1_puzzles_in_pgn_san_split_add_nl.json"  # 평가 데이터 파일 경로
    print(f"====== Run Testing on {test_data_path} ======\n\n")
    run_testing(TRAINING_CFG, test_data_path)
