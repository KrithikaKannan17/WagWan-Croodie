"""
Test script for Phase 3: Verify trained model can be loaded and used.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from chess import Board
from src.utils.model import ChessModel
from src.utils.board_encoder import board_to_tensor_torch
from src.utils.move_mapper import MoveMapper


def test_model_loading():
    """Test that trained model can be loaded."""
    print("Testing model loading...")
    
    # Look for model in parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(parent_dir, "chess_model.pth")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  ✓ Loaded checkpoint from {model_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    except FileNotFoundError:
        print(f"  ✗ Model file {model_path} not found. Run training first.")
        return False
    
    # Create model and load weights
    model = ChessModel(hidden_size=128, num_hidden_layers=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("  ✓ Model loaded successfully")
    return True, model, checkpoint.get('move_mapper')


def test_model_inference(model, move_mapper):
    """Test that model can make predictions."""
    print("\nTesting model inference...")
    
    # Create a test board
    board = Board()
    board_tensor = board_to_tensor_torch(board)
    
    # Get legal moves
    legal_moves = list(board.generate_legal_moves())
    print(f"  Legal moves: {len(legal_moves)}")
    
    # Run inference
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value: {value.item():.4f}")
    
    # Convert to move probabilities
    if move_mapper:
        move_probs = move_mapper.get_move_probabilities(
            policy_logits[0].numpy(), legal_moves
        )
        print(f"  Move probabilities computed for {len(move_probs)} legal moves")
        
        # Check that probabilities sum to ~1
        prob_sum = sum(move_probs.values())
        print(f"  Sum of probabilities: {prob_sum:.4f}")
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to ~1, got {prob_sum}"
    else:
        print("  Note: Move mapper not found in checkpoint")
    
    print("  ✓ Model inference test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Testing: Model Training & Loading")
    print("=" * 60)
    
    try:
        result = test_model_loading()
        if isinstance(result, tuple):
            success, model, move_mapper = result
            if success:
                test_model_inference(model, move_mapper)
                
                print("\n" + "=" * 60)
                print("✓ All Phase 3 tests passed!")
                print("=" * 60)
        else:
            print("\n✗ Model loading failed")
            exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

