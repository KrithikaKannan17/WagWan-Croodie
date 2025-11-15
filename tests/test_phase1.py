"""
Test script for Phase 1: Verify board encoding and model inference.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess import Board
from src.utils.board_encoder import board_to_tensor, board_to_tensor_torch
from src.utils.model import ChessModel
import torch

def test_board_encoding():
    """Test that board encoding works correctly."""
    print("Testing board encoding...")
    
    # Test with starting position
    board = Board()
    tensor = board_to_tensor(board)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Expected: (8, 8, 12)")
    assert tensor.shape == (8, 8, 12), f"Expected (8, 8, 12), got {tensor.shape}"
    
    # Check that we have pieces in the right places
    # White pawns should be on rank 1 (index 6 in our encoding)
    white_pawns = tensor[:, :, 0]  # Channel 0 = White Pawns
    print(f"White pawns on rank 1 (index 6): {white_pawns[6, :].sum()}")
    assert white_pawns[6, :].sum() == 8, "Should have 8 white pawns on rank 1"
    
    # Black pawns should be on rank 6 (index 1 in our encoding)
    black_pawns = tensor[:, :, 6]  # Channel 6 = Black Pawns
    print(f"Black pawns on rank 6 (index 1): {black_pawns[1, :].sum()}")
    assert black_pawns[1, :].sum() == 8, "Should have 8 black pawns on rank 6"
    
    print("✓ Board encoding test passed!")
    return True

def test_model_inference():
    """Test that model can perform inference."""
    print("\nTesting model inference...")
    
    # Create model
    model = ChessModel(hidden_size=128, num_hidden_layers=2)
    model.eval()  # Set to evaluation mode
    
    # Create a board and encode it
    board = Board()
    board_tensor = board_to_tensor_torch(board)
    
    print(f"Input tensor shape: {board_tensor.shape}")
    print(f"Expected: (1, 12, 8, 8)")
    assert board_tensor.shape == (1, 12, 8, 8), f"Expected (1, 12, 8, 8), got {board_tensor.shape}"
    
    # Run inference
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        move_probs, value_out = model.predict(board_tensor)
    
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Move probabilities shape: {move_probs.shape}")
    print(f"Value output: {value_out.item():.4f}")
    
    assert policy_logits.shape == (1, 256), f"Expected (1, 256), got {policy_logits.shape}"
    assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"
    assert move_probs.shape == (1, 256), f"Expected (1, 256), got {move_probs.shape}"
    
    # Check that probabilities sum to ~1
    prob_sum = move_probs.sum().item()
    print(f"Sum of move probabilities: {prob_sum:.4f}")
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to ~1, got {prob_sum}"
    
    print("✓ Model inference test passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Phase 1 Testing: Board Encoding & Model Inference")
    print("=" * 50)
    
    try:
        test_board_encoding()
        test_model_inference()
        
        print("\n" + "=" * 50)
        print("✓ All Phase 1 tests passed!")
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

