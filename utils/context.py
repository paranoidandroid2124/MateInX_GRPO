import chess
from typing import Optional
import warnings

def simple_context_updater(fen: str, llm_move: str, env_move: Optional[str]):
    """
    보드 상태를 업데이트하면서 유효성을 검증합니다.
    
    Args:
        fen: 현재 보드 상태 (FEN notation)
        llm_move: LLM이 제안한 이동 (SAN format)
        env_move: 환경(상대방)의 이동 (SAN format, optional)
    
    Returns:
        str: 업데이트된 보드 상태 (FEN notation)
    """
    try:
        board = chess.Board(fen)
        
        # 보드 상태 유효성 검증
        if not board.is_valid():
            warnings.warn(f"Invalid board state: {fen}")
            return fen  # 원래 상태 반환
            
    except ValueError as e:
        warnings.warn(f"Failed to parse FEN: {fen}, error: {e}")
        return fen
    
    # 이동 처리
    for move_str in [llm_move, env_move]:
        if move_str:
            try:
                #move_obj = chess.Move.from_uci(move_str)
                move_obj = board.parse_san(move_str)
                
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                    print(f"✅ Applied move: {move_str}")
                else:
                    print(f"❌ Illegal move {move_str} on board {board.fen()}")
                    # 잘못된 이동은 무시하고 계속 진행
                    
            except ValueError as e:
                print(f"❌ Invalid san move format: {move_str}, error: {e}")
                continue
                
        # 게임 종료 상태 체크
        if board.is_game_over():
            print(f"🏁 Game over: {board.outcome()}")
            break
    
    return board.fen()
