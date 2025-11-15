"""
Review script for Phase 3: Inspect trained model and test predictions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from chess import Board, Move
from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor_torch
from src.utils.move_mapper import MoveMapper


def load_model_and_mapper(model_path=None):
    """Load the trained model and move mapper."""
    if model_path is None:
        # Look for model in parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, "chess_model.pth")
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with same architecture
    model = ChessModel(hidden_size=128, num_hidden_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    move_mapper = checkpoint.get('move_mapper')
    if move_mapper is None:
        print("  Warning: Move mapper not found in checkpoint, creating new one")
        move_mapper = MoveMapper()
    
    print(f"  ✓ Model loaded from {model_path}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, move_mapper


def test_starting_position(model, move_mapper):
    """Test model on starting position."""
    print("\n" + "=" * 60)
    print("Test 1: Starting Position")
    print("=" * 60)
    
    board = Board()
    board_tensor = board_to_tensor_torch(board)
    legal_moves = list(board.generate_legal_moves())
    
    print(f"Position: Starting position")
    print(f"Legal moves: {len(legal_moves)}")
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    print(f"Position evaluation: {value.item():.4f} (closer to 1 = better for white)")
    
    # Get move probabilities
    move_probs = move_mapper.get_move_probabilities(
        policy_logits[0].numpy(), legal_moves
    )
    
    # Sort by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 predicted moves:")
    for i, (move, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {move.uci()}: {prob:.4f} ({prob*100:.2f}%)")
    
    return sorted_moves[0][0]  # Return top move


def test_custom_position(model, move_mapper, fen=None):
    """Test model on a custom position."""
    print("\n" + "=" * 60)
    print("Test 2: Custom Position")
    print("=" * 60)
    
    if fen is None:
        # Test a common opening position (after 1.e4 e5)
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    
    board = Board(fen=fen)
    board_tensor = board_to_tensor_torch(board)
    legal_moves = list(board.generate_legal_moves())
    
    print(f"Position FEN: {fen}")
    print(f"Legal moves: {len(legal_moves)}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    # Adjust value based on whose turn it is
    if not board.turn:  # Black's turn
        value = -value
    
    print(f"Position evaluation: {value.item():.4f} (from current player's perspective)")
    
    # Get move probabilities
    move_probs = move_mapper.get_move_probabilities(
        policy_logits[0].numpy(), legal_moves
    )
    
    # Sort by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 predicted moves:")
    for i, (move, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {move.uci()}: {prob:.4f} ({prob*100:.2f}%)")
    
    return sorted_moves[0][0]


def test_endgame_position(model, move_mapper):
    """Test model on an endgame position."""
    print("\n" + "=" * 60)
    print("Test 3: Endgame Position")
    print("=" * 60)
    
    # Simple endgame: King and Queen vs King
    fen = "8/8/8/8/8/8/4K3/4Q3 w - - 0 1"
    
    board = Board(fen=fen)
    board_tensor = board_to_tensor_torch(board)
    legal_moves = list(board.generate_legal_moves())
    
    print(f"Position: King + Queen vs King (winning for White)")
    print(f"Legal moves: {len(legal_moves)}")
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    print(f"Position evaluation: {value.item():.4f} (should be positive for White)")
    
    # Get move probabilities
    move_probs = move_mapper.get_move_probabilities(
        policy_logits[0].numpy(), legal_moves
    )
    
    # Sort by probability
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 predicted moves:")
    for i, (move, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {move.uci()}: {prob:.4f} ({prob*100:.2f}%)")


def compare_with_random(model, move_mapper, num_positions=5):
    """Compare model predictions with random selection."""
    print("\n" + "=" * 60)
    print("Test 4: Model vs Random (Consistency Check)")
    print("=" * 60)
    
    board = Board()
    
    for i in range(num_positions):
        # Make some random moves to get different positions
        if i > 0:
            legal = list(board.generate_legal_moves())
            if legal:
                board.push(np.random.choice(legal))
        
        board_tensor = board_to_tensor_torch(board)
        legal_moves = list(board.generate_legal_moves())
        
        if not legal_moves:
            break
        
        with torch.no_grad():
            policy_logits, value = model(board_tensor)
        
        move_probs = move_mapper.get_move_probabilities(
            policy_logits[0].numpy(), legal_moves
        )
        
        # Get model's top choice
        model_move = max(move_probs.items(), key=lambda x: x[1])[0]
        model_prob = move_probs[model_move]
        
        # Random choice would have probability 1/len(legal_moves)
        random_prob = 1.0 / len(legal_moves)
        
        print(f"Position {i+1}: {len(legal_moves)} legal moves")
        print(f"  Model top move: {model_move.uci()} (prob: {model_prob:.4f})")
        print(f"  Random prob: {random_prob:.4f}")
        print(f"  Model confidence: {model_prob/random_prob:.2f}x random")
        print()


def show_model_statistics(model):
    """Show statistics about the model."""
    print("\n" + "=" * 60)
    print("Model Statistics")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Count parameters by layer type
    linear_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            linear_params += param.numel()
    
    print(f"Linear layer parameters: {linear_params:,}")
    print(f"Model architecture: MLP with policy and value heads")


def main():
    print("=" * 60)
    print("Phase 3 Review: Trained Model Analysis")
    print("=" * 60)
    
    try:
        # Load model
        model, move_mapper = load_model_and_mapper()
        
        # Show model statistics
        show_model_statistics(model)
        
        # Test on various positions
        test_starting_position(model, move_mapper)
        test_custom_position(model, move_mapper)
        test_endgame_position(model, move_mapper)
        compare_with_random(model, move_mapper)
        
        print("\n" + "=" * 60)
        print("Review Complete!")
        print("=" * 60)
        print("\nWhat to look for:")
        print("1. Model should assign different probabilities to different moves")
        print("2. Value predictions should make sense (positive for winning positions)")
        print("3. Model should be more confident than random (higher probabilities)")
        print("4. Top moves should be reasonable chess moves")
        print("\nNote: Since model was trained on random games, it may not play")
        print("optimally, but it should show learning (non-uniform probabilities).")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please train the model first by running:")
        print("  python train.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

