"""
Simple Tactical Position Generator for Chess Training
Generates positions with clear tactical themes: forks, pins, hanging pieces, checkmates
"""

import argparse
import numpy as np
import chess
import random
from typing import List, Tuple

from src.utils.board_encoder import board_to_tensor
from src.utils.move_mapper import MoveMapper


def simple_evaluate(board: chess.Board) -> float:
    """Simple material-based evaluation."""
    if board.is_checkmate():
        return -10000 if board.turn else 10000
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    
    return score


def find_tactical_move(board: chess.Board) -> Tuple[chess.Move, str, float]:
    """Find a tactical move in the current position."""
    best_move = None
    best_score = -99999
    tactic_type = "position"
    
    for move in board.legal_moves:
        # Check if it's a capture (piece on destination square)
        is_capture = board.piece_at(move.to_square) is not None
        
        # Make move
        board.push(move)
        
        # Check for checkmate
        if board.is_checkmate():
            board.pop()
            return move, "checkmate", 1.0
        
        # Check for hanging pieces or captures
        score = -simple_evaluate(board)
        
        # Bonus for captures
        if is_capture:
            score += 2
            if score > best_score:
                best_score = score
                best_move = move
                tactic_type = "capture"
        
        # Check for checks
        if board.is_check():
            score += 1
            if score > best_score:
                best_score = score
                best_move = move
                tactic_type = "check"
        
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = move
    
    if best_move is None:
        best_move = random.choice(list(board.legal_moves))
    
    # Normalize score to [-1, 1]
    value = max(-1.0, min(1.0, best_score / 10.0))
    
    return best_move, tactic_type, value


def generate_tactical_position() -> Tuple[np.ndarray, int, float]:
    """
    Generate a single tactical position.
    
    Returns:
        board_tensor: (8, 8, 12) board encoding
        move_index: Best tactical move index
        value: Position evaluation [-1, 1]
    """
    board = chess.Board()
    move_mapper = MoveMapper()
    
    # Play 10-30 random moves to get a mid-game position
    num_moves = random.randint(10, 30)
    for _ in range(num_moves):
        if board.is_game_over():
            board = chess.Board()
            continue
        
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            board.push(move)
    
    # Skip if game over
    if board.is_game_over():
        return generate_tactical_position()  # Retry
    
    # Find tactical move
    tactical_move, tactic_type, value = find_tactical_move(board)
    
    # Encode board
    board_tensor = board_to_tensor(board)
    
    # Get move index
    move_index = move_mapper.get_move_index(tactical_move)
    
    # Adjust value for current player
    if not board.turn:  # Black to move
        value = -value
    
    return board_tensor, move_index, value


def generate_tactical_dataset(num_positions: int, output_path: str):
    """Generate a dataset of tactical positions."""
    print(f"\n{'='*70}")
    print(f"🎯 TACTICAL POSITION GENERATOR")
    print(f"{'='*70}")
    print(f"Generating {num_positions} tactical positions...")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")
    
    boards = []
    policy_targets = []
    value_targets = []
    
    move_mapper = MoveMapper()
    
    for i in range(num_positions):
        try:
            board_tensor, move_index, value = generate_tactical_position()
            
            # Create policy target (one-hot for best move)
            policy_vector = np.zeros(256, dtype=np.float32)
            policy_vector[move_index] = 1.0
            
            boards.append(board_tensor)
            policy_targets.append(policy_vector)
            value_targets.append(value)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_positions} positions...", flush=True)
        
        except Exception as e:
            print(f"Error generating position {i}: {e}")
            continue
    
    # Convert to numpy arrays
    boards = np.array(boards, dtype=np.float32)
    policy_targets = np.array(policy_targets, dtype=np.float32)
    value_targets = np.array(value_targets, dtype=np.float32)
    
    # Convert boards from (N, 8, 8, 12) to (N, 12, 8, 8)
    boards = np.transpose(boards, (0, 3, 1, 2))
    
    # Save
    np.savez_compressed(
        output_path,
        boards=boards,
        policy_targets=policy_targets,
        value_targets=value_targets
    )
    
    print(f"\n{'='*70}")
    print(f"✅ TACTICAL DATASET GENERATED")
    print(f"{'='*70}")
    print(f"  Positions: {len(boards)}")
    print(f"  Boards shape: {boards.shape}")
    print(f"  Policy targets shape: {policy_targets.shape}")
    print(f"  Value targets shape: {value_targets.shape}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*70}\n")
    
    print(f"Value distribution:")
    print(f"  Winning positions (>0.3): {np.sum(value_targets > 0.3)}")
    print(f"  Equal positions (-0.3 to 0.3): {np.sum(np.abs(value_targets) <= 0.3)}")
    print(f"  Losing positions (<-0.3): {np.sum(value_targets < -0.3)}")


def main():
    parser = argparse.ArgumentParser(description='Generate tactical chess positions')
    parser.add_argument('--num-positions', type=int, default=1000,
                       help='Number of positions to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='tactical_data.npz',
                       help='Output file path (default: tactical_data.npz)')
    
    args = parser.parse_args()
    
    generate_tactical_dataset(args.num_positions, args.output)
    
    print("\n🎯 Next steps:")
    print(f"  1. Train on tactical data:")
    print(f"     modal run train_modal_selfplay.py \\")
    print(f"       --data-file {args.output.split('/')[-1]} \\")
    print(f"       --model-input chess_model_sp_v2.pth \\")
    print(f"       --model-output chess_model_tactical.pth \\")
    print(f"       --epochs 10")
    print(f"\n  2. Download and evaluate:")
    print(f"     modal run train_modal_selfplay.py --model-output chess_model_tactical.pth --download")
    print(f"     python -m src.evaluation.test_model_comparison \\")
    print(f"       --model1 chess_model_sp_v2.pth \\")
    print(f"       --model2 chess_model_tactical.pth \\")
    print(f"       --games 20\n")


if __name__ == "__main__":
    main()

