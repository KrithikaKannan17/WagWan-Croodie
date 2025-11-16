"""
Test two models against each other to compare strength.
"""

import torch
import chess
from src.utils.model import ChessModel
from src.utils.move_mapper import MoveMapper
from src.utils.mcts import MCTS
import time


def load_model(model_path, device="cpu"):
    """Load a model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_residual_blocks = checkpoint.get('num_residual_blocks', 6)
        channels = checkpoint.get('channels', 64)
        state_dict = checkpoint['model_state_dict']
    else:
        num_residual_blocks = 6
        channels = 64
        state_dict = checkpoint
    
    model = ChessModel(num_residual_blocks=num_residual_blocks, channels=channels, dropout=0.0)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def play_game(model1, model2, move_mapper, mcts_sims=50, max_moves=100, verbose=False):
    """
    Play one game between two models.

    Returns:
        1 if model1 wins, -1 if model2 wins, 0 if draw
    """
    board = chess.Board()
    mcts1 = MCTS(model1, move_mapper, num_simulations=mcts_sims)
    mcts2 = MCTS(model2, move_mapper, num_simulations=mcts_sims)

    moves = 0

    while not board.is_game_over() and moves < max_moves:
        # Choose MCTS based on whose turn it is
        mcts = mcts1 if board.turn == chess.WHITE else mcts2

        # Get best move with temperature for exploration
        best_move, move_probs = mcts.search(board)

        # Add some randomness in early game to prevent loops
        if moves < 20:
            import random
            import numpy as np
            # Sample from top moves with temperature
            moves_list = list(move_probs.keys())
            probs_list = np.array([move_probs[m] for m in moves_list])
            probs_list = probs_list ** 0.5  # Temperature = 2.0 (more random)
            probs_list = probs_list / probs_list.sum()
            best_move = np.random.choice(moves_list, p=probs_list)

        if verbose:
            print(f"  Move {moves + 1}: {best_move.uci()}")

        board.push(best_move)
        moves += 1

        # Check for draw by repetition (only after some moves)
        if moves > 10 and board.can_claim_draw():
            if verbose:
                print(f"  Draw claimed at move {moves}")
            return 0

    # Determine outcome
    if verbose:
        print(f"  Game ended after {moves} moves")
        print(f"  Game over: {board.is_game_over()}")
        print(f"  Checkmate: {board.is_checkmate()}")
        print(f"  Stalemate: {board.is_stalemate()}")

    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    else:
        return 0  # Draw


def test_models(model1_path, model2_path, num_games=10, mcts_sims=50, device="cpu"):
    """
    Test two models against each other.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        num_games: Number of games to play
        mcts_sims: MCTS simulations per move
        device: Device to run on
    """
    print("="*70)
    print("🎮 Model vs Model Testing")
    print("="*70)
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Games: {num_games}, MCTS sims: {mcts_sims}")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)
    move_mapper = MoveMapper()
    print("✓ Models loaded")
    
    # Play games
    results = []
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    print(f"\nPlaying {num_games} games...")
    print("-"*70)
    
    for i in range(num_games):
        start_time = time.time()

        # Verbose for first game only
        verbose = (i == 0)

        # Alternate colors
        if i % 2 == 0:
            result = play_game(model1, model2, move_mapper, mcts_sims, verbose=verbose)
        else:
            result = -play_game(model2, model1, move_mapper, mcts_sims, verbose=verbose)
        
        results.append(result)
        
        if result == 1:
            model1_wins += 1
            outcome = "Model 1 wins"
        elif result == -1:
            model2_wins += 1
            outcome = "Model 2 wins"
        else:
            draws += 1
            outcome = "Draw"
        
        elapsed = time.time() - start_time
        print(f"Game {i+1}/{num_games} | {outcome} | Time: {elapsed:.1f}s")
    
    # Print results
    print("="*70)
    print("📊 Results:")
    print("="*70)
    print(f"Model 1 wins: {model1_wins} ({model1_wins/num_games*100:.1f}%)")
    print(f"Model 2 wins: {model2_wins} ({model2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("="*70)
    
    if model1_wins > model2_wins:
        print("🏆 Model 1 is stronger!")
    elif model2_wins > model1_wins:
        print("🏆 Model 2 is stronger!")
    else:
        print("🤝 Models are equally strong!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test two chess models against each other")
    parser.add_argument("--model1", type=str, required=True, help="Path to first model")
    parser.add_argument("--model2", type=str, required=True, help="Path to second model")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    test_models(args.model1, args.model2, args.games, args.mcts_sims, args.device)

