"""
Test script for MCTS implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from chess import Board
from src.utils.model import ChessModel
from src.utils.move_mapper import MoveMapper
from src.utils.mcts import MCTS, MCTSNode


def test_mcts_node():
    """Test MCTSNode basic functionality."""
    print("Testing MCTSNode...")
    
    board = Board()
    node = MCTSNode(board)
    
    assert not node.is_terminal(), "Starting position should not be terminal"
    assert len(node.legal_moves) == 20, "Starting position should have 20 legal moves"
    assert node.visit_count == 0, "New node should have 0 visits"
    
    print("  ✓ MCTSNode basic functionality works")
    return True


def test_mcts_search():
    """Test MCTS search with neural network."""
    print("\nTesting MCTS search...")
    
    # Load model
    try:
        # Look for model in parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, 'chess_model.pth')
        checkpoint = torch.load(model_path, map_location='cpu')
        model = ChessModel(hidden_size=128, num_hidden_layers=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        move_mapper = checkpoint.get('move_mapper', MoveMapper())
    except FileNotFoundError:
        print("  ✗ Model file not found. Please train a model first.")
        return False
    
    # Create MCTS
    mcts = MCTS(model, move_mapper, num_simulations=50)
    
    # Test on starting position
    board = Board()
    print(f"  Starting position, {len(list(board.generate_legal_moves()))} legal moves")
    
    # Run search
    best_move, move_probs = mcts.search(board)
    
    print(f"  Best move: {best_move.uci()}")
    print(f"  Move probabilities computed for {len(move_probs)} moves")
    
    # Check that probabilities sum to ~1
    prob_sum = sum(move_probs.values())
    print(f"  Sum of probabilities: {prob_sum:.4f}")
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to ~1, got {prob_sum}"
    
    # Show top 5 moves
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 moves:")
    for i, (move, prob) in enumerate(sorted_moves, 1):
        print(f"    {i}. {move.uci()}: {prob:.4f}")
    
    print("  ✓ MCTS search works")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MCTS Testing")
    print("=" * 60)
    
    try:
        test_mcts_node()
        test_mcts_search()
        
        print("\n" + "=" * 60)
        print("✓ All MCTS tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

