"""
Watch a model play against itself or a random player.
"""

import torch
import chess
from src.utils.model import ChessModel
from src.utils.move_mapper import MoveMapper
from src.utils.mcts import MCTS
import random


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


def watch_game(model_path, mcts_sims=50, opponent="self", max_moves=100, device="cpu"):
    """
    Watch a model play a game.
    
    Args:
        model_path: Path to model
        mcts_sims: MCTS simulations per move
        opponent: "self" or "random"
        max_moves: Maximum moves before draw
        device: Device to run on
    """
    print("="*70)
    print("👀 Watch Model Play")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Opponent: {opponent}")
    print(f"MCTS sims: {mcts_sims}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path, device)
    move_mapper = MoveMapper()
    mcts = MCTS(model, move_mapper, num_simulations=mcts_sims)
    print("✓ Model loaded")
    
    # Play game
    board = chess.Board()
    move_count = 0
    
    print("\n" + "="*70)
    print("Game Start")
    print("="*70)
    print(board)
    print()
    
    while not board.is_game_over() and move_count < max_moves:
        # Determine who plays
        if opponent == "self" or (opponent == "random" and board.turn == chess.WHITE):
            # Model plays
            best_move, move_probs = mcts.search(board)
            
            # Get top 3 moves
            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"Move {move_count + 1} ({'White' if board.turn else 'Black'}):")
            print(f"  Chosen: {best_move.uci()} (prob: {move_probs[best_move]:.3f})")
            print(f"  Top 3: {[(m.uci(), f'{p:.3f}') for m, p in sorted_moves]}")
            
            # Get position value
            if hasattr(mcts, 'root'):
                value = mcts.root.get_value()
                print(f"  Position value: {value:.3f}")
            
        else:
            # Random plays
            legal_moves = list(board.legal_moves)
            best_move = random.choice(legal_moves)
            print(f"Move {move_count + 1} ({'White' if board.turn else 'Black'}): {best_move.uci()} (random)")
        
        board.push(best_move)
        move_count += 1
        
        print()
        print(board)
        print()
        
        # Check for draw
        if board.can_claim_draw():
            print("Draw by repetition/50-move rule!")
            break
        
        # Pause for readability
        if move_count % 5 == 0:
            input("Press Enter to continue...")
    
    # Print result
    print("="*70)
    print("Game Over")
    print("="*70)
    
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"🏆 Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("🤝 Stalemate!")
    elif board.can_claim_draw():
        print("🤝 Draw by repetition/50-move rule!")
    elif move_count >= max_moves:
        print(f"🤝 Draw by move limit ({max_moves} moves)!")
    else:
        print("🤝 Draw!")
    
    print(f"Total moves: {move_count}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch a chess model play")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--opponent", type=str, default="self", choices=["self", "random"], 
                        help="Opponent type")
    parser.add_argument("--max-moves", type=int, default=100, help="Max moves before draw")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    watch_game(args.model, args.mcts_sims, args.opponent, args.max_moves, args.device)

