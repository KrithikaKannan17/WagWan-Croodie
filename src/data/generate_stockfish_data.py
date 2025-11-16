"""
Generate high-quality training data using Stockfish engine.
This creates much better training data than random games.
"""

import argparse
import chess
import chess.engine
import numpy as np
from src.utils.board_encoder import board_to_tensor
from src.utils.data_collection import save_training_data
import os

def generate_stockfish_data(num_games=100, stockfish_path=None, depth=10):
    """
    Generate training data using Stockfish to find best moves.
    
    Args:
        num_games: Number of games to generate
        stockfish_path: Path to Stockfish executable (auto-detect if None)
        depth: Stockfish search depth (higher = better but slower)
    """
    # Try to find Stockfish
    if stockfish_path is None:
        possible_paths = [
            "stockfish",  # If in PATH
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "stockfish.exe",
        ]
        for path in possible_paths:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(path)
                stockfish_path = path
                print(f"✓ Found Stockfish at: {path}")
                break
            except:
                continue
        else:
            print("✗ Stockfish not found!")
            print("  Download from: https://stockfishchess.org/download/")
            print("  Or specify path with --stockfish-path")
            return None, None, None
    else:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    print(f"\n📊 Generating {num_games} games with Stockfish (depth {depth})...")
    print("This will take a while but produces much better training data!\n")
    
    all_board_states = []
    all_moves = []
    all_outcomes = []
    
    for game_idx in range(num_games):
        if (game_idx + 1) % 10 == 0:
            print(f"  Progress: {game_idx + 1}/{num_games} games...")
        
        board = chess.Board()
        game_positions = []
        game_moves = []
        
        # Play game with Stockfish
        move_count = 0
        max_moves = 150
        
        while move_count < max_moves and not board.is_game_over():
            # Get best move from Stockfish
            result = engine.play(board, chess.engine.Limit(depth=depth))
            move = result.move
            
            # Store position and move
            game_positions.append(board_to_tensor(board))
            game_moves.append(move.uci())
            
            board.push(move)
            move_count += 1
        
        # Determine outcome
        if board.is_checkmate():
            outcome = 1.0 if not board.turn else -1.0
        else:
            outcome = 0.0
        
        # Add to dataset
        for state, move in zip(game_positions, game_moves):
            all_board_states.append(state)
            all_moves.append(move)
            all_outcomes.append(outcome)
    
    engine.quit()
    
    board_states = np.array(all_board_states)
    outcomes = np.array(all_outcomes)
    
    print(f"\n✓ Generated {len(board_states)} training positions")
    print(f"  Wins: {np.sum(outcomes > 0)}")
    print(f"  Draws: {np.sum(outcomes == 0)}")
    print(f"  Losses: {np.sum(outcomes < 0)}")
    
    return board_states, all_moves, outcomes


def main():
    parser = argparse.ArgumentParser(description='Generate training data using Stockfish')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games to generate (default: 100)')
    parser.add_argument('--depth', type=int, default=10,
                       help='Stockfish search depth (default: 10)')
    parser.add_argument('--stockfish-path', type=str, default=None,
                       help='Path to Stockfish executable')
    parser.add_argument('--output', type=str, default='datasets/training_data_stockfish.npz',
                       help='Output filename')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("High-Quality Training Data Generation with Stockfish")
    print("=" * 60)
    
    result = generate_stockfish_data(
        num_games=args.num_games,
        stockfish_path=args.stockfish_path,
        depth=args.depth
    )
    
    if result[0] is not None:
        board_states, moves, outcomes = result
        save_training_data(board_states, moves, outcomes, args.output)
        
        print("\n" + "=" * 60)
        print(f"✓ High-quality training data saved to {args.output}")
        print(f"  Total positions: {len(board_states)}")
        print("\nNext step: Train your model with this data!")
        print(f"  python -m src.training.train_phase2 --data {args.output} --epochs 30")
        print("=" * 60)


if __name__ == "__main__":
    main()

