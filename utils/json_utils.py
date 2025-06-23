from typing import List, Dict, Any, Optional
from .patterns import JSON_PATTERN

def extract_json(model_output_str: str) -> Optional[str]:
    match = JSON_PATTERN.findall(model_output_str)
    return match[-1] if match else None

def get_candidate_info(candidates: List[Dict[str, Any]], move: str) -> Dict[str, str]:
    """
    후보 수 리스트에서 특정 수(move)에 대한 정보를 찾아 반환합니다.
    
    Args:
        candidates: 후보 수 리스트 (모델이 제안한 여러 후보 수들)
        move: 정보를 찾고 싶은 특정 수 (UCI 포맷, 예: "e2e4")
        
    Returns:
        Dict: 찾아낸 수와 그 수에 대한 설명을 담은 딕셔너리
    """
    # 1. next_move와 정확히 일치하는 후보 찾기
    matching_candidates = [
        c for c in candidates 
        if isinstance(c, dict) and c.get("move") == move
    ]
    
    # 2. 일치하는 후보가 있으면 첫 번째 것 사용 (일반적으로 가장 높은 점수의 후보)
    if matching_candidates:
        info = matching_candidates[0]
        return {
            "move": move, 
            "reasoning": info.get("reasoning", "")
        }
    
    # 3. 일치하는 후보가 없으면 빈 reasoning 반환
    return {"move": move, "reasoning": ""}
