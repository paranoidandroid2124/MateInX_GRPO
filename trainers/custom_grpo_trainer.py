import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import wandb
import os
import json
from datetime import datetime
import chess
from copy import deepcopy

from trl import GRPOTrainer, GRPOConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from accelerate.utils import is_peft_model

from utils.json_utils import get_candidate_info


@dataclass
class TrajectoryExperience:
    initial_query_input_ids: torch.Tensor
    initial_query_attention_mask: torch.Tensor
    initial_response_input_ids: List[torch.Tensor]
    initial_response_attention_mask: List[torch.Tensor]
    initial_response_log_prob: List[torch.Tensor] # List of scalar tensors (sequence log_probs)
    cumulative_trajectory_reward: torch.Tensor  # shape: [num_generations]

# ---------------------------------------------------------------------------
#  Custom Trainer
# ---------------------------------------------------------------------------

class CustomChessGRPOTrainer(GRPOTrainer):
    """GRPO trainer tailored for chess‑puzzle trajectories using *ChessReward*."""

    # ------------------------------------------------------------------
    #  Init
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: GRPOConfig,
        puzzle_reward_model: Any,
        puzzle_dataset_for_rollout: List[Dict[str, Any]],
        prompt_formatter: Callable[[str, str, List[Dict[str, Any]]], Tuple[List[Dict[str, str]], str, str]],
        context_updater: Callable[[str, str, Optional[str]], str],
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        reward_funcs: Optional[List[Callable]] = None,
        train_dataset: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            args=config,
            reward_funcs=reward_funcs or [lambda completions, **_: [0.0] * len(completions)],
            optimizers=(optimizer, lr_scheduler),
            train_dataset=train_dataset,
            **kwargs,
        )

        self.tokenizer = tokenizer
        self.puzzle_reward_model = puzzle_reward_model
        self.puzzle_dataset_for_rollout = puzzle_dataset_for_rollout
        self.prompt_formatter = prompt_formatter
        self.context_updater = context_updater
        self.training_config = training_config or {}
        self.puzzle_step = 0
        self.generation_step = 0 

        max_new_tokens = self.training_config.get("max_new_tokens", 128)
        
        gen_defaults = {
            "max_new_tokens": max_new_tokens,
            "min_length": -1,
            "top_k": 0.0, 
            "top_p": 1.0, 
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        self._gen_kwargs = {**gen_defaults, **getattr(self.args, "generation_kwargs", {})}

    # ------------------------------------------------------------------
    #  Log‑prob helper (token‑level)
    # ------------------------------------------------------------------

    def _get_logprobs(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        generated_ids: torch.Tensor, 
        *,
        model_instance: Optional[PreTrainedModel] = None,
    ) -> torch.Tensor: 
        """Return per‑token logp tensor aligned to *generated_ids* (pad masked)."""
        model = model_instance or self.accelerator.unwrap_model(self.model)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        effective_prompt_len = input_ids.size(1) - generated_ids.size(1)
        
        relevant_logits = logits[:, effective_prompt_len-1 : input_ids.size(1)-1, :]
        
        max_gen_len = relevant_logits.size(1)
        clamped_generated_ids = generated_ids[:, :max_gen_len]
        
        log_probs_dist = F.log_softmax(relevant_logits, dim=-1) 
        
        token_lp = torch.gather(log_probs_dist, 2, clamped_generated_ids.unsqueeze(-1)).squeeze(-1)
        
        padding_mask = clamped_generated_ids.eq(self.tokenizer.pad_token_id)
        token_lp = token_lp.masked_fill(padding_mask, 0.0)
        
        del logits, relevant_logits, log_probs_dist 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return token_lp

    # ------------------------------------------------------------------
    #  Batched Generation utility
    # ------------------------------------------------------------------
    def _generate_batch(self, prompts: List[str]) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Generates responses for a batch of prompts.
        Returns:
            - List[str]: list of response texts
            - torch.Tensor: batch of response_ids (B, max_new_tokens)
            - torch.Tensor: batch of response_masks (B, max_new_tokens)
            - torch.Tensor: batch of sequence_log_probs (B,)
            - List[float]: list of generation times per prompt
        """
        if not prompts:
            return [], torch.empty(0, dtype=torch.long, device=self.accelerator.device), \
                   torch.empty(0, dtype=torch.long, device=self.accelerator.device), \
                   torch.empty(0, dtype=torch.float, device=self.accelerator.device), []

        toks = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.accelerator.device)
        
        current_gen_kwargs = {**self._gen_kwargs}
        current_gen_kwargs["num_return_sequences"] = 1 
        current_gen_kwargs["output_scores"] = True 
        current_gen_kwargs["return_dict_in_generate"] = True

        params = {
            **current_gen_kwargs, 
            "input_ids": toks["input_ids"], 
            "attention_mask": toks["attention_mask"],
        }
        
        start_time = time.time()
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        with torch.no_grad():
            out_generated = model.generate(**params)
        model.train()
        end_time = time.time()
        
        generation_time_per_prompt = (end_time - start_time) / len(prompts) if prompts else 0
        generation_times = [generation_time_per_prompt] * len(prompts)

        prompt_lengths = toks["attention_mask"].sum(dim=1)
        max_gen_len_config = current_gen_kwargs["max_new_tokens"]

        response_ids_list = []
        for i in range(out_generated.sequences.size(0)):
            prompt_len_i = prompt_lengths[i]
            if prompt_len_i > out_generated.sequences.size(1):
                 res_ids = torch.tensor([], dtype=torch.long, device=self.accelerator.device)
            else:
                res_ids = out_generated.sequences[i, prompt_len_i:]
            
            pad_len = max_gen_len_config - res_ids.size(0)
            if pad_len > 0:
                res_ids = F.pad(res_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            response_ids_list.append(res_ids)

        response_ids_batch = torch.stack(response_ids_list)
        response_texts = [self.tokenizer.decode(r_ids, skip_special_tokens=True) for r_ids in response_ids_batch]
        
        full_generated_sequences_mask = out_generated.sequences.ne(self.tokenizer.pad_token_id).long()

        token_lp_batch = self._get_logprobs(
            out_generated.sequences, 
            full_generated_sequences_mask, 
            response_ids_batch, 
            model_instance=model
        )
        
        seq_logp_sum_batch = token_lp_batch.sum(dim=-1)
        resp_mask_batch = response_ids_batch.ne(self.tokenizer.pad_token_id).long()
        
        return response_texts, response_ids_batch, resp_mask_batch, seq_logp_sum_batch, generation_times

    # ------------------------------------------------------------------
    #  Batched Trajectory execution (rollout)
    # ------------------------------------------------------------------
    def _execute_batched_trajectories(self, puzzle_dict: Dict[str, Any], themes: str, num_trajectories: int) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[Dict[str, Any]], List[int]]:
        
        initial_board_ctx = puzzle_dict["initial_context"]
        solution_moves = puzzle_dict["solution_moves"]
        total_llm_steps = puzzle_dict["total_llm_steps"]

        # 응답 저장을 위한 로그 디렉토리 및 파일 설정
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        # 각 퍼즐(롤아웃)마다 고유한 로그 파일 생성 또는 타임스탬프로 구분
        # 여기서는 단순화를 위해 하나의 파일에 추가하지만, 실제로는 파일 관리가 필요할 수 있음
        log_file_path = os.path.join(log_dir, f"model_responses.jsonl")
        
        # 로그 버퍼 초기화 (각 trajectory의 로그를 모음)
        # 이 버퍼는 현재 퍼즐의 모든 trajectory, 모든 스텝에 대한 로그를 담게 됨
        current_puzzle_log_buffer: List[str] = []
        last_responses_lengths: List[int] = []
        
        # 세션 정보 저장 (퍼즐 단위)
        session_info = {
            "type": "session_info",
            "puzzle_id": self.puzzle_step, # 현재 처리 중인 퍼즐 ID
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "initial_fen": initial_board_ctx,
            "total_llm_steps_for_puzzle": total_llm_steps,
            "num_trajectories_for_puzzle": num_trajectories,
            "themes": themes if themes else ""
        }
        current_puzzle_log_buffer.append(json.dumps(session_info))
                
        current_board_ctxs = [initial_board_ctx] * num_trajectories
        trajectory_states = []
        for i in range(num_trajectories):
            trajectory_states.append(deepcopy({ 
                "trajectory_id": i, # 각 trajectory 구분용 ID
                "cs": 0, "is": 0, "current_step_in_puzzle": 0,
                "total_steps_in_puzzle": total_llm_steps, "puzzle_ended": False,
                "previous_llm_moves_and_reasonings": [],
                "episode_move": 0, "episode_is": 0, "episode_cs": 0, "episode_format": 0
            }))

        cumulative_rewards = torch.zeros(num_trajectories, device=self.accelerator.device, dtype=torch.float32)
        
        first_response_ids_all_trajs = [torch.empty(0, dtype=torch.long, device=self.accelerator.device)] * num_trajectories
        first_response_masks_all_trajs = [torch.empty(0, dtype=torch.long, device=self.accelerator.device)] * num_trajectories
        first_response_log_probs_all_trajs = [torch.tensor(0.0, device=self.accelerator.device)] * num_trajectories

        batched_initial_query_ids: Optional[torch.Tensor] = None
        batched_initial_query_mask: Optional[torch.Tensor] = None
        
        active_trajectory_indices = list(range(num_trajectories))

        for step_num in range(total_llm_steps):
            if not active_trajectory_indices:
                break

            prompts_for_active_batch = []
            original_indices_for_current_batch: List[int] = [] 

            for original_idx in active_trajectory_indices:
                board_ctx = current_board_ctxs[original_idx]
                prev_moves = trajectory_states[original_idx]["previous_llm_moves_and_reasonings"]
                
                messages, fen_nl, san_legal_moves = self.prompt_formatter(board_ctx, themes, prev_moves)
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompt += fen_nl + " " + san_legal_moves + '</think>'
                
                prompts_for_active_batch.append(prompt)
                original_indices_for_current_batch.append(original_idx)

            if not prompts_for_active_batch:
                break 

            resp_texts_b, resp_ids_b, resp_masks_b, logps_b, generation_times_b = self._generate_batch(prompts_for_active_batch)

            if step_num == 0:
                if prompts_for_active_batch:
                    encoded_first_prompt = self.tokenizer(prompts_for_active_batch[0], return_tensors="pt", padding=False, truncation=True).to(self.accelerator.device)
                    batched_initial_query_ids = encoded_first_prompt["input_ids"].squeeze(0) 
                    batched_initial_query_mask = encoded_first_prompt["attention_mask"].squeeze(0)
                
                for i_batch, original_idx in enumerate(original_indices_for_current_batch):
                    first_response_ids_all_trajs[original_idx] = resp_ids_b[i_batch]
                    first_response_masks_all_trajs[original_idx] = resp_masks_b[i_batch]
                    first_response_log_probs_all_trajs[original_idx] = logps_b[i_batch] 

            next_step_active_indices: List[int] = [] 

            for i_batch, original_idx in enumerate(original_indices_for_current_batch):
                resp_txt = resp_texts_b[i_batch]
                current_step_response_length = len(resp_txt)
                step_generation_time = generation_times_b[i_batch]
                current_traj_state = trajectory_states[original_idx]
                current_board_for_this_step = current_board_ctxs[original_idx] 

                target_move_idx = current_traj_state["current_step_in_puzzle"] * 2
                if target_move_idx >= len(solution_moves):
                    warnings.warn(f"Trajectory {original_idx}, Step {step_num}: Target move index {target_move_idx} out of bounds. Ending.")
                    current_traj_state["puzzle_ended"] = True
                    # No further processing for this trajectory in this step if it ends here
                else: # Only proceed if target_move_idx is valid
                    current_target_move = solution_moves[target_move_idx]

                    step_reward, next_state_dict, parsed_info = self.puzzle_reward_model.calculate_step_reward_and_next_state(
                        resp_txt, current_target_move, current_traj_state, current_board_for_this_step
                    )
                    cumulative_rewards[original_idx] += step_reward
                    trajectory_states[original_idx] = next_state_dict 
                    
                    next_move_parsed = parsed_info.get("next_move", "unknown") if parsed_info else "unknown"
                    
                    print(f'🧩 Puzzle {self.puzzle_step + 1} | Traj {original_idx + 1} | Step {step_num + 1}/{total_llm_steps} | Target: {current_target_move} | Model: {next_move_parsed} | ✅: {next_move_parsed == current_target_move} | Reward: {step_reward:.3f}')

                    temp_board_for_turn_info = chess.Board(current_board_for_this_step)
                    previous_fen_for_log = current_board_for_this_step
                    previous_turn_for_log = "White" if temp_board_for_turn_info.turn == chess.WHITE else "Black"
                    
                    reasoning_log = ""
                    if parsed_info and 'candidates' in parsed_info:
                        candidate_info = get_candidate_info(parsed_info.get("candidates", []), next_move_parsed)
                        reasoning_log = candidate_info.get("reasoning", "")
                        if next_move_parsed == current_target_move:
                            move_info_log = {
                                "move": next_move_parsed, "reasoning": reasoning_log,
                                "reasoning_short": candidate_info.get("reasoning", "Correct strategic move"),
                                "predicted_line": json.dumps(parsed_info.get("predicted_main_line", [])),
                                "previous_fen": previous_fen_for_log, "previous_turn": previous_turn_for_log, "was_correct": True
                            }
                        else:
                            move_info_log = {
                                "move": next_move_parsed, "reasoning": "", "reasoning_short": "Incorrect move attempted",
                                "predicted_line": "[]", "previous_fen": previous_fen_for_log,
                                "previous_turn": previous_turn_for_log, "was_correct": False
                            }
                    else: # JSON 파싱 실패 또는 candidates 없음
                        move_info_log = {
                            "move": next_move_parsed, "reasoning": "", "reasoning_short": "Parse error or no candidates",
                            "predicted_line": "[]", "previous_fen": previous_fen_for_log,
                            "previous_turn": previous_turn_for_log, "was_correct": False
                        }
                    trajectory_states[original_idx]["previous_llm_moves_and_reasonings"].append(move_info_log)

                    # 스텝별 로그 항목 구성
                    response_log_entry = {
                        "type": "model_response_step",
                        "step_in_puzzle": step_num + 1, # 1-indexed step
                        "total_steps_in_puzzle": total_llm_steps,
                        "step_time_seconds": round(step_generation_time, 3),
                        "current_fen_before_move": current_board_for_this_step,
                        "target_move": current_target_move,
                        "model_next_move": next_move_parsed,
                        "is_correct": next_move_parsed == current_target_move,
                        "step_reward": round(step_reward, 4),
                        "cumulative_reward_for_trajectory": round(cumulative_rewards[original_idx].item(), 4),
                        "model_full_response": parsed_info if parsed_info else resp_txt, # 파싱된 정보 또는 원본 텍스트
                        "reasoning_from_response": reasoning_log
                    }
                    current_puzzle_log_buffer.append(json.dumps(response_log_entry))


                # Trajectory 종료 처리 및 Wandb 로깅
                if trajectory_states[original_idx]["puzzle_ended"]:
                    last_responses_lengths.append(current_step_response_length)
                    try:
                        wandb.log({
                            f"episode/generation_step": self.generation_step, 
                            f"episode/format": trajectory_states[original_idx]["episode_format"],
                            f"episode/move": trajectory_states[original_idx]["episode_move"],
                            f"episode/total_reward": cumulative_rewards[original_idx].item() / total_llm_steps,
                            f"episode/last_response_length": current_step_response_length,
                            f"episode/correct_streak": trajectory_states[original_idx]["episode_cs"],
                            f"episode/incorrect_streak": trajectory_states[original_idx]["episode_is"],
                        })
                        self.generation_step +=1
                    except (ImportError, Exception):
                        pass 
                elif target_move_idx < len(solution_moves): # 유효한 다음 스텝이 있을 경우에만 보드 업데이트 및 활성 목록 추가
                    llm_solution_move_idx = (trajectory_states[original_idx]["current_step_in_puzzle"] - 1) * 2
                    env_solution_move_idx = llm_solution_move_idx + 1

                    if llm_solution_move_idx >= len(solution_moves):
                        warnings.warn(f"Trajectory {original_idx}, Step {step_num}: LLM solution move index {llm_solution_move_idx} out of bounds for board update. Ending.")
                        trajectory_states[original_idx]["puzzle_ended"] = True
                        # continue # 이미 위에서 target_move_idx 체크로 걸러지거나, 여기서 종료됨
                    else:
                        llm_move_from_solution = solution_moves[llm_solution_move_idx]
                        env_move_from_solution = solution_moves[env_solution_move_idx] if env_solution_move_idx < len(solution_moves) else None
                        
                        current_board_ctxs[original_idx] = self.context_updater(
                            current_board_for_this_step, 
                            llm_move_from_solution, 
                            env_move_from_solution
                        )
                        next_step_active_indices.append(original_idx)
            
            active_trajectory_indices = next_step_active_indices
        
        # 현재 퍼즐의 모든 trajectory 로그를 파일에 기록
        try:
            if current_puzzle_log_buffer:
                with open(log_file_path, "a") as f: # 'a' 모드로 추가
                    for log_line in current_puzzle_log_buffer:
                        f.write(log_line + "\n")
        except IOError as e:
            warnings.warn(f"Failed to write logs to {log_file_path}: {e}")


        # Consolidate results for GRPO
        final_traj_results_for_grpo: List[Dict[str, Any]] = []
        for i in range(num_trajectories):
            final_traj_results_for_grpo.append({
                "ids": first_response_ids_all_trajs[i],
                "mask": first_response_masks_all_trajs[i],
                "logp": first_response_log_probs_all_trajs[i], 
                "score": cumulative_rewards[i].item() 
            })
        
        if batched_initial_query_ids is None:
            warnings.warn("Batched initial query IDs are None. This might indicate all trajectories failed early.")
            if num_trajectories > 0 : 
                 dummy_prompt_text = self.tokenizer.bos_token or " " 
                 encoded_dummy = self.tokenizer(dummy_prompt_text, return_tensors="pt", padding=False, truncation=True).to(self.accelerator.device)
                 batched_initial_query_ids = encoded_dummy["input_ids"].squeeze(0)
                 batched_initial_query_mask = encoded_dummy["attention_mask"].squeeze(0)

        return batched_initial_query_ids, batched_initial_query_mask, final_traj_results_for_grpo, last_responses_lengths

    # ------------------------------------------------------------------
    #  Rollout generator for GRPO
    # ------------------------------------------------------------------

    def generate_rollout_for_grpo(
        self, batch_raw: List[Dict[str, Any]] 
    ) -> List[TrajectoryExperience]:
        experiences: List[TrajectoryExperience] = []
        for puzzle_raw_item in batch_raw: 
            puzzle_as_dict = {
                "initial_context": puzzle_raw_item["FEN"],
                "solution_moves": puzzle_raw_item["MovesSAN"], 
                "total_llm_steps": (len(puzzle_raw_item["MovesSAN"]) + 1) // 2,
            }
            themes_str = puzzle_raw_item.get("Themes", "")

            initial_query_ids, initial_query_mask, list_of_traj_result_dicts, model_responses_lengths = \
                self._execute_batched_trajectories(
                    puzzle_as_dict, 
                    themes_str, 
                    self.args.num_generations 
                )

            if initial_query_ids is None or not list_of_traj_result_dicts:
                warnings.warn(f"Skipping puzzle {self.puzzle_step} due to no valid initial query or trajectory results. FEN: {puzzle_as_dict['initial_context']}")
                self.puzzle_step += 1 # 실패한 퍼즐도 카운트하여 다음 퍼즐 ID가 밀리지 않도록 함
                continue
            
            exp = TrajectoryExperience(
                initial_query_input_ids=initial_query_ids,       
                initial_query_attention_mask=initial_query_mask, 
                initial_response_input_ids=[res["ids"] for res in list_of_traj_result_dicts],      
                initial_response_attention_mask=[res["mask"] for res in list_of_traj_result_dicts],
                initial_response_log_prob=[res["logp"] for res in list_of_traj_result_dicts],      
                cumulative_trajectory_reward=torch.tensor(
                    [res["score"] for res in list_of_traj_result_dicts], device=self.accelerator.device, dtype=torch.float32
                ), 
            )

            if self.accelerator.is_main_process:
                if exp.cumulative_trajectory_reward.numel() > 0:
                    avg_reward = float(torch.mean(exp.cumulative_trajectory_reward).item())
                    std_reward = float(torch.std(exp.cumulative_trajectory_reward).item())
                    min_reward = float(torch.min(exp.cumulative_trajectory_reward).item())
                    max_reward = float(torch.max(exp.cumulative_trajectory_reward).item())

                    wandb.log({
                        "mate/puzzle_step": self.puzzle_step, # wandb에 기록되는 스텝은 실제 처리 완료된 퍼즐 기준
                        "mate/avg_reward_per_puzzle": avg_reward,
                        "mate/std_reward_per_puzzle": std_reward,
                        "mate/min_reward_per_puzzle": min_reward,
                        "mate/max_reward_per_puzzle": max_reward,
                        "mate/avg_response_lengths": sum(model_responses_lengths) / len(model_responses_lengths),
                    })
                else:
                    warnings.warn("Cumulative trajectory reward tensor is empty. Skipping wandb logging for this puzzle.")
            
            self.puzzle_step += 1 # 다음 퍼즐 ID를 위해 증가

            experiences.append(exp)
        return experiences

    # ------------------------------------------------------------------
    #  Batch preparation for GRPO step
    # ------------------------------------------------------------------

    def _prepare_data_for_step(self, exps: List[TrajectoryExperience]) -> Dict[str, Any]:
        if not exps:
            return {}

        def _flatten(attr: str): 
            return [val for e in exps for val in getattr(e, attr)]

        queries = []
        query_masks = []
        for e in exps:
            num_gens_for_this_exp = len(e.initial_response_input_ids)
            queries.extend([e.initial_query_input_ids] * num_gens_for_this_exp)
            query_masks.extend([e.initial_query_attention_mask] * num_gens_for_this_exp)
        
        completions = _flatten("initial_response_input_ids")
        comp_masks = _flatten("initial_response_attention_mask")
        
        rewards_list = [e.cumulative_trajectory_reward for e in exps]
        if not rewards_list: return {}
        rewards = torch.cat(rewards_list) 

        if rewards.numel() == 0: return {}
        
        if rewards.numel() > 0 and rewards.numel() % self.args.num_generations == 0:
             grouped_rewards = rewards.view(-1, self.args.num_generations)
             adv = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
             adv = adv / (grouped_rewards.std(dim=1, keepdim=True) + 1e-8) 
             adv = adv.reshape(-1) 
        else:
            warnings.warn(f"Rewards count ({rewards.numel()}) not a multiple of num_generations ({self.args.num_generations}). Using unnormalized rewards as advantages.")
            adv = rewards 

        pad_to_dev = lambda seqs, val: pad_sequence(seqs, batch_first=True, padding_value=val).to(self.accelerator.device)
        
        prompt_ids = pad_to_dev(queries, self.tokenizer.pad_token_id)
        prompt_mask = pad_to_dev(query_masks, 0) 
        comp_ids = pad_to_dev(completions, self.tokenizer.pad_token_id)
        comp_mask = pad_to_dev(comp_masks, 0)

        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        full_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        old_tok_lp = self._get_per_token_logps(self.accelerator.unwrap_model(self.model), full_ids, full_mask, comp_ids.size(1))

        data = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": comp_ids,
            "completion_mask": comp_mask,
            "old_per_token_logps": old_tok_lp, 
            "advantages": adv.to(self.accelerator.device),
        }

        if self.args.beta > 0.0:
            model_ref = self.ref_model or self.accelerator.unwrap_model(self.model) 
            ctx_mgr = None
            if self.ref_model is None and is_peft_model(model_ref):
                ctx_mgr = model_ref.disable_adapter(); ctx_mgr.__enter__() # type: ignore
            
            with torch.no_grad():
                ref_tok_lp = self._get_per_token_logps(model_ref, full_ids, full_mask, comp_ids.size(1))
            
            if ctx_mgr:
                ctx_mgr.__exit__(None, None, None)
            data["ref_per_token_logps"] = ref_tok_lp
        return data

    # ------------------------------------------------------------------
    #  Optimize step
    # ------------------------------------------------------------------

    def step(self, experiences: List[TrajectoryExperience], **kwargs):  # type: ignore[override]
        if not experiences:
            if self.accelerator.is_main_process:
                warnings.warn("No experiences in step; skipping update.")
            return None

        batch = self._prepare_data_for_step(experiences)
        if not batch or batch["advantages"].numel() == 0:
            if self.accelerator.is_main_process:
                warnings.warn("Empty advantages; skipping update.")
            return None

        loss = self.compute_loss(self.model, batch, return_outputs=False)
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients and self.args.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        self.optimizer.step(); self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.detach()
    
    
    
