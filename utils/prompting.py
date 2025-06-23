import json
import chess
from chess import Board

from typing import List, Dict, Any

def format_board(board_str: str) -> str:
    result = "\n   +------------------------+\n"
    # 입력 문자열의 앞뒤 공백 제거 및 개행 기준으로 분리
    rows = board_str.strip().split("\n")
    # 각 행의 기물들을 공백 기준으로 분리
    tokens = [row.split() for row in rows]
    for i in range(8):
        result += f" {8-i} | " + "  ".join(tokens[i]) + " |\n"
    result += "   +------------------------+\n"
    result += "     a  b  c  d  e  f  g  h\n"
    return result

def describe_materials(board):
    idx_to_file = {0: 'a', 
                   1: 'b', 
                   2: 'c',
                   3: 'd',
                   4: 'e',
                   5: 'f',
                   6: 'g',
                   7: 'h'}
    ranks = board.split('\n')
    locations_str = ""
    for rank_idx, rank in enumerate(ranks):
        for file_idx, material in enumerate(rank.split()):
            file = idx_to_file[file_idx]
            if material == '.':
                continue
            elif material.islower():
                tmp = "Black "
            elif material.isupper():
                tmp = "White "

            if material.lower() == 'p':
                tmp += f"pawn at {file}{8-rank_idx}. "
            elif material.lower() == 'k':
                tmp += f"king at {file}{8-rank_idx}. "
            elif material.lower() == 'r':
                tmp += f"rook at {file}{8-rank_idx}. "
            elif material.lower() == 'n':
                tmp += f"knight at {file}{8-rank_idx}. "
            elif material.lower() == 'b':
                tmp += f"bishop at {file}{8-rank_idx}. "
            elif material.lower() == 'q':
                tmp += f"queen at {file}{8-rank_idx}. "
            locations_str += tmp
    
    return locations_str

def castle_status(board: Board) -> str:
    """
    주어진 체스 보드 객체에서 캐슬링 가능 여부를 16가지 경우에 맞춰
    자연어 문자열로 설명합니다.

    Args:
        board: python-chess의 Board 객체

    Returns:
        캐슬링 가능 여부를 설명하는 문자열
    """

    # 각 캐슬링 권한 확인
    k_white = board.has_kingside_castling_rights(chess.WHITE)
    q_white = board.has_queenside_castling_rights(chess.WHITE)
    k_black = board.has_kingside_castling_rights(chess.BLACK)
    q_black = board.has_queenside_castling_rights(chess.BLACK)

    # 가능한 캐슬링 목록 생성
    available_castles = []
    if k_white:
        available_castles.append("White kingside")
    if q_white:
        available_castles.append("White queenside")
    if k_black:
        available_castles.append("Black kingside")
    if q_black:
        available_castles.append("Black queenside")

    count = len(available_castles)

    # 경우의 수에 따라 문자열 생성
    if count == 0:
        return "Both players have exhausted their castles."
    elif count == 1:
        return f"{available_castles[0]} is the only castle move available."
    elif count == 4:
        return "All castling moves are available for both players."
    elif count == 2:
        return f"{available_castles[0]} and {available_castles[1]} castles are available."
    else: # count == 3
        return f"{available_castles[0]}, {available_castles[1]} and {available_castles[2]} castles are available."


def fen_to_nl(fen):
    """FEN board to nl.
    example:
        input: 
            2rk1bnr/N3pp2/1p2b3/1pp3Pp/5q2/P1NP4/R1P1P1P1/2QK1B2 w - - 1 23
        output:
            The current state of the chess board is: Black rook at c8, black king at d8, black bishop at f8, black knight at g8 and black rook at h8. White knight at a7, black pawn at e7 and black pawn at f7. Black pawn at b6 and black bishop at e6. Black pawn at b5, black pawn at c5, white pawn at g5 and black pawn at h5. Black queen at f4. White pawn at a3, white knight at c3 and white pawn at d3. White rook at a2, white pawn at c2, white pawn at e2 and white pawn at g2. White queen at c1, white king at d1 and white bishop at f1. 
            Both players have exhausted their castles. 
            Half move number is 1. 
            Full move number is 23. 
            White to move.
    """
    nl = "The current state of the chess board is: "
    board = Board(fen)
    material_locations = describe_materials(str(board))
    nl += material_locations
    nl += castle_status(board)
    halfmove_clock, fullmove_number= fen.split()[4], fen.split()[5]
    nl += f" Half move number is {halfmove_clock}."
    nl += f" Full move number is {fullmove_number}."
    player_turn = "White" if fen.split()[1]=='w' else "Black"
    nl += f" {player_turn} to move."

    return nl

def chess_prompt_formatter(fen: str, themes: str, prev: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    체스 퍼즐 프롬프트 포맷터 - 고정된 퓨샷 예시 + 이전 정답 이동 활용
    
    반환: 메시지 리스트 (시스템 + 퓨샷 예시들 + 이전 성공 이동 + 현재 질문)
    """
    # 시스템 메시지 - 역할 정의 및 지침
    system_prompt = """You are an expert chess grandmaster with deep knowledge of chess tactics, strategies, and patterns. Your task is to find the optimal move given the chess puzzle position.
Board Representation:
A FEN record contains six fields, each separated by a space. The fields are as follows:

Piece placement data: Each rank is described, starting with rank 8 and ending with rank 1, with a "/" between each one; within each rank, the contents of the squares are described in order from the a-file to the h-file. Each piece is identified by a single letter taken from the standard English names in algebraic notation (pawn = "P", knight = "N", bishop = "B", rook = "R", queen = "Q" and king = "K"). White pieces are designated using uppercase letters ("PNBRQK"), while black pieces use lowercase letters ("pnbrqk"). A set of one or more consecutive empty squares within a rank is denoted by a digit from "1" to "8", corresponding to the number of squares.
Active color: "w" means that White is to move; "b" means that Black is to move.
Castling availability: If neither side has the ability to castle, this field uses the character "-". Otherwise, this field contains one or more letters: "K" if White can castle kingside, "Q" if White can castle queenside, "k" if Black can castle kingside, and "q" if Black can castle queenside. A situation that temporarily prevents castling does not prevent the use of this notation.
En passant target square: This is a square over which a pawn has just passed while moving two squares; it is given in algebraic notation. If there is no en passant target square, this field uses the character "-". This is recorded regardless of whether there is a pawn in position to capture en passant. An updated version of the spec has since made it so the target square is recorded only if a legal en passant capture is possible, but the old version of the standard is the one most commonly used.
Halfmove clock: The number of halfmoves since the last capture or pawn advance, used for the fifty-move rule.
Fullmove number: The number of the full moves. It starts at 1 and is incremented after Black's move.

Here is the FEN for the starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
And after the move e2e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1

To determine the optimal move and generate the required JSON, follow this thought process:
1. **Understand the Position**: Thoroughly analyze the board state using the FEN string. Pay close attention to piece placement, threats, and strategic opportunities for both sides.
2. **Identify and Evaluate Candidates**: Based on your understanding, identify several strong candidate moves. For each candidate, formulate a concise reasoning explaining its strategic or tactical merit.
3. **Select the Best Move**: From your evaluated candidates, determine the single best move. This move should be reflected as `next_move` in your JSON output, and its reasoning should be present in the `candidates` list.
4. **Formulate the JSON Response**: Finally, structure your analysis into a single, complete JSON object as detailed below.

Your Response Structure:
Your response MUST be a valid JSON object in the following format:
{
  "candidates": [ {"move": "[SAN format]", "reasoning": "[A concise summary of the main idea for this candidate]"} ],
  "predicted_main_line": ["[SAN format]", "[SAN format]", ...],
  "next_move": "[SAN format]"
}

Important - SAN Move Format:
All chess moves in the JSON MUST use the SAN (Universal Chess Interface) format.
Examples: 'e4' (pawn move), 'Nf3' (knight move), 'O-O' (White kingside castling), 'Rae1' (specific rook to e1), 'Qxe4+' (queen captures on e4 with check), 'e8=Q#' (pawn promotes to queen with checkmate).

Chess Notation Guide (for SAN):
- Pawn moves: written as the destination square (e.g., 'e4').
- Piece moves: use an uppercase letter for the piece (K, Q, R, B, N) followed by the destination square (e.g., 'Nf3' means a knight moves to f3).
- Captures: add 'x' before the destination square (e.g., 'Nxe5' means a knight captures on e5).
- Disambiguation: if two identical pieces can move to the same square, specify the originating file or rank (e.g., 'Nbd2', 'R1e2').
- Castling: written as 'O-O' for kingside and 'O-O-O' for queenside.
- Checks: add '+' at the end (e.g., 'Qh5+').
- Checkmates: add '#' at the end (e.g., 'Qxf7#').
- Pawn promotions: use '=' followed by the piece promoted to (e.g., 'e8=Q' promotes a pawn to a queen), and combine with check or mate symbols if applicable (e.g., 'e8=Q+', 'e8=Q#').
"""

    # 고정된 퓨샷 예시들
    EXAMPLE_FENS = [
        "r1q1kb1r/p1p2ppp/1pn1pn2/8/2BP4/1QN1PP2/PP1B1P1P/R3K2R w KQkq - 0 10",
        "r2qkb1r/ppp1pppp/5n2/3Pn3/4P1b1/2N2N2/PP3PPP/R1BQKB1R w KQkq - 1 8",
        "rnb1k2r/pp3ppp/2pP1qn1/4N3/3PP3/1QP5/P4PPP/R1B1KB1R w KQkq - 1 11"
    ]

    EXAMPLE_THEMES = [
        "opening, sacrifice, advantage, crushing",
        "sacrifice, attackingF2F7", 
        "opening, trappedPiece"
    ]

    EXAMPLE_RESPONSES = [
        """{
  "candidates": [
    {"move": "Bb5", "reasoning": "Directly attacks Nc6, tactically strong."},
    {"move": "Qa4", "reasoning": "Aims to pin Nc6."}
  ],
  "predicted_main_line": ["Bb5", "Qd7", "Bxc6", "Qxc6", "Ne4"],
  "next_move": "Bb5"
}""",
        """{
  "candidates": [
    {"move": "Nxe5", "reasoning": "Sacrifices knight but sets up powerful attack."},
    {"move": "Bb5", "reasoning": "Check first and gain positional advantage."}
  ],
  "predicted_main_line": ["Nxe5", "Bxd1", "Bb5", "c6", "dxc6"],
  "next_move": "Nxe5"
}""",
        """{
  "candidates": [
    {"move": "Bg5", "reasoning": "Creates deadly pin against queen."},
    {"move": "d5b6", "reasoning": "Captures material but releases tension."}
  ],
  "predicted_main_line": ["Bg5", "Qe6", "Bc4", "Qd7", "Bxf7"],
  "next_move": "Bg5"
}"""
    ]

    # 현재 위치 정보 구성
    turn = "Unknown"
    try:
        board_obj = chess.Board(fen)
        turn = "White" if board_obj.turn == chess.WHITE else "Black"
    except Exception as e:
        print(f"Error processing FEN: {e}")
    
    # 현재 질문 구성
    current_user_prompt = f"Find the best move given this chess position.\n\nFEN: {fen}\nPlayer to move: {turn}"
    if themes:
        current_user_prompt += f"\nPuzzle Themes: {themes}"
    fen_understanding_nl = fen_to_nl(fen) # 사고과정 토큰 절약을 위한 자연어 보드 이해
    fen_legal_moves = ""
    #fen_legal_moves = "The legal moves from the current chessboard position are: "+",".join(board_obj.san(m).rstrip('+#') for m in board_obj.legal_moves) # 사고과정 토큰 절약을 위한
    
    # 메시지 구성: 시스템 + 퓨샷 예시들 + 이전 성공 이동들 + 현재 질문
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 고정된 퓨샷 예시들 추가
    for i in range(len(EXAMPLE_FENS)):
        # 예시 사용자 질문
        example_user_content = f"Find the best move given this chess position.\n\nFEN: {EXAMPLE_FENS[i]}\nPlayer to move: White"
        if EXAMPLE_THEMES[i]:
            example_user_content += f"\nPuzzle Themes: {EXAMPLE_THEMES[i]}"
        
        # 예시 응답
        example_assistant_content = EXAMPLE_RESPONSES[i]
        
        messages.append({"role": "user", "content": example_user_content})
        messages.append({"role": "assistant", "content": example_assistant_content})
    
    # 🎯 이전 정답 이동들을 대화 히스토리로 추가 (정답만)
    if prev:
        correct_moves = [move for move in prev if move.get("was_correct", False)]
        
        for move in correct_moves:
            if move.get("reasoning"):  # reasoning이 있는 경우만
                # 이전 스텝의 사용자 질문 재구성
                prev_user_prompt = f"Find the best move given this chess position.\n\nFEN: {move['previous_fen']}\nPlayer to move: {move['previous_turn']}"
                
                # 이전 스텝의 성공적인 응답 재구성
                prev_assistant_response = f"""{{
  "candidates": [
    {{"move": "{move['move']}", "reasoning": "{move['reasoning']}"}}
  ],
  "predicted_main_line": {move['predicted_line']},
  "next_move": "{move['move']}"
}}"""
                
                messages.append({"role": "user", "content": prev_user_prompt})
                messages.append({"role": "assistant", "content": prev_assistant_response})
    
    # 현재 실제 질문 추가
    messages.append({"role": "user", "content": current_user_prompt}) 
    
    
    return messages, fen_understanding_nl, fen_legal_moves
