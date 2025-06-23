import json
import wandb
from typing import Dict, Optional, Any, Tuple
import chess
from utils.patterns import UCI_PATTERN
from utils.json_utils import extract_json, get_candidate_info

class ChessReward:
    _PENALTY_KEYS = {
        "json_extraction": -1.0,
        "json_decoding": -1.0,
        "missing_candidates": -1.0,
        "wrong_type_candidates": -1.0,
        "missing_predicted_main_line": -1.0,
        "wrong_type_predicted_main_line": -1.0,
        "missing_next_move": -1.0,
        "wrong_type_next_move": -1.0,
        "candidate_not_dict": -1.0,
        "candidate_missing_move": -1.0,
        "candidate_missing_reasoning": -1.0,
        "uci_format": -1.0,

    }

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        **penalty_overrides: float,
    ) -> None:
        # Allow caller to override individual penalties while keeping defaults.
        self.penalties: Dict[str, float] = self._PENALTY_KEYS | penalty_overrides

        cfg = config or {}
        
        self.format_pass_base_reward = cfg.get("format_pass_base_reward", 0.1)
        self.move_correct_base_reward = cfg.get("move_correct_base_reward", 0.1)
        self.move_incorrect_base_penalty = cfg.get("move_incorrect_base_penalty", -0.1)
        self.correct_streak_multiplier = cfg.get("correct_streak_multiplier", 1.1)
        self.incorrect_streak_multiplier = cfg.get("incorrect_streak_multiplier", 0.9)
        self.check_uci = cfg.get("check_uci", False)
        self.reward_uci_format_correct = cfg.get("reward_uci_format_correct", 0.1)

        self.trajectory_completion_reward = cfg.get("trajectory_completion_reward", 1.0)
        self.trajectory_failure_penalty = cfg.get("trajectory_failure_penalty", -1.0)
        self.partial_move_from_reward = cfg.get("partial_move_from_reward", 0.05)
        self.partial_move_to_reward =  cfg.get("partial_move_to_reward", 0.03)
        self.partial_move_file_reward = cfg.get("partial_move_file_reward", 0.003)
        self.partial_move_rank_reward = cfg.get("partial_move_rank_reward", 0.003)
        self.legal_move_reward = cfg.get("legal_move_reward", 0.15)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_uci(move: str) -> bool:
        return bool(UCI_PATTERN.match(move))

    def _compute_format_reward(
        self, model_output_str: str
    ) -> Tuple[float, Optional[Dict[str, Any]], bool, Optional[str]]:
        """Verify output structure & apply format‑related reward / penalty."""
        # Extract JSON
        json_str = extract_json(model_output_str)
        if json_str is None:
            return self.penalties["json_extraction"], None, False, "json_extraction"

        # Decode JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return self.penalties["json_decoding"], None, False, "json_decoding"

        # Required keys & expected types
        required = {
            "candidates": list,
            "predicted_main_line": list,
            "next_move": str,
        }
        for key, expected in required.items():
            if key not in parsed:
                return self.penalties[f"missing_{key}"], parsed, False, f"missing_{key}"
            if not isinstance(parsed[key], expected):
                return self.penalties[f"wrong_type_{key}"], parsed, False, f"wrong_type_{key}"

        for cand in parsed["candidates"]:
            if not isinstance(cand, dict):
                return self.penalties["candidate_not_dict"], parsed, False, "candidate_not_dict"
            if "move" not in cand:
                return self.penalties["candidate_missing_move"], parsed, False, "candidate_missing_move"
            if "reasoning" not in cand:
                return (
                    self.penalties["candidate_missing_reasoning"],
                    parsed,
                    False,
                    "candidate_missing_reasoning",
                )

        # Passed all structural checks
        reward = self.format_pass_base_reward

        # SAN format으로 바뀌었기 때문에 uci format panelty는 주석처리 @TODO: valid SAN에 대한 보상
        # UCI format check
        # if self.check_uci:
        #     reward += (
        #         self.reward_uci_format_correct
        #         if self._is_valid_uci(parsed["next_move"])
        #         else self.penalties["uci_format"]
        #     )
        return reward, parsed, True, None

    def calculate_step_reward_and_next_state(
        self,
        model_output_str: str,
        actual_next_move: str,
        state: Dict[str, Any],
        fen: str
    ) -> Tuple[float, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Compute reward for one LLM step and return updated puzzle *state* and parsed info."""
        reward, parsed, fmt_ok, fail_reason = self._compute_format_reward(model_output_str)
        next_state = state.copy()

        # If format invalid, terminate puzzle early
        if not fmt_ok:
            next_state.update(
                {
                    "reason_for_ending": f"F({fail_reason})",
                    "final_reward_for_step": reward,
                    "cs": 0,
                    "is": 0
                }
            )
            next_state["episode_format"] += reward

            next_state["current_step_in_puzzle"] = next_state.get("current_step_in_puzzle", 0) + 1

            next_state["puzzle_ended"] = (
                next_state["current_step_in_puzzle"]
                >= next_state.get("total_steps_in_puzzle", float("inf"))
            )


            return reward, next_state, None

        llm_move = parsed["next_move"]
        correct_streak = next_state.get("cs", 0)
        incorrect_streak = next_state.get("is", 0)

        # Move evaluation -------------------------------------------------
        if llm_move == actual_next_move:
            move_reward = self.move_correct_base_reward * (
                self.correct_streak_multiplier ** correct_streak
            )

            next_state["episode_cs"] =correct_streak
            next_state.update({"cs": correct_streak + 1, "is": 0})
             

        else:
            move_reward = self.move_incorrect_base_penalty * (
                self.incorrect_streak_multiplier ** incorrect_streak
            )

            # partial rewards
            # for partial reward, use uci format.
            
            try:
                board = chess.Board(fen)
                llm_move = board.uci(board.parse_san(llm_move)) #change to uci format
                move_reward += self.legal_move_reward # give legal move reward
                actual_next_move = board.uci(board.parse_san(actual_next_move)) #change to uci format
                if len(llm_move) >= 4 and len(actual_next_move) >= 4:
                    if llm_move[0:2]== actual_next_move[0:2]:
                        move_reward += self.partial_move_from_reward 
                    elif (llm_move[0] == actual_next_move[0]): 
                        move_reward += self.partial_move_file_reward
                    elif (llm_move[1] == actual_next_move[1]):
                        move_reward += self.partial_move_rank_reward
                    
                    if llm_move[2:4] == actual_next_move[2:4]:
                        move_reward += self.partial_move_to_reward
                    
                    elif (llm_move[2] == actual_next_move[2]):
                        move_reward += self.partial_move_file_reward
            
                    elif (llm_move[3] == actual_next_move[3]): 
                        move_reward += self.partial_move_rank_reward
            except:
                print("illegal san given the board context. give elementwise reward instead of uci partial reward.")
                for i in range(min(len(llm_move), len(actual_next_move))):
                    if llm_move[i] == actual_next_move[i]:
                        move_reward += self.partial_move_file_reward

            next_state["episode_is"] = incorrect_streak
            next_state.update({"cs": 0, "is": incorrect_streak + 1})


        # Aggregate rewards ----------------------------------------------
        reward += move_reward
        next_state["final_reward_for_step"] = reward
        next_state["episode_move"] += move_reward

        next_state["current_step_in_puzzle"] = next_state.get("current_step_in_puzzle", 0) + 1

        next_state["puzzle_ended"] = (
            next_state["current_step_in_puzzle"]
            >= next_state.get("total_steps_in_puzzle", float("inf"))
        )
        return reward, next_state, parsed

