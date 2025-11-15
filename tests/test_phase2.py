"""
Test script for Phase 2: Verify training data generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.data_collection import load_training_data, generate_training_data, save_training_data


def test_data_generation():
    """Test that data generation works correctly."""
    print("Testing data generation...")
    
    # Generate a small dataset
    board_states, moves, outcomes = generate_training_data(num_games=3, max_moves=20)
    
    print(f"  Generated {len(board_states)} positions")
    print(f"  Number of moves: {len(moves)}")
    print(f"  Number of outcomes: {len(outcomes)}")
    
    # Verify shapes
    assert len(board_states) == len(moves) == len(outcomes), "All lists should have same length"
    assert len(board_states) > 0, "Should generate at least some positions"
    
    # Verify board state shape
    first_state = board_states[0]
    assert first_state.shape == (8, 8, 12), f"Expected (8, 8, 12), got {first_state.shape}"
    
    # Verify outcomes are valid
    for outcome in outcomes:
        assert outcome in [-1.0, 0.0, 1.0], f"Outcome should be -1, 0, or 1, got {outcome}"
    
    # Verify moves are valid
    for move in moves:
        assert isinstance(move, str) or hasattr(move, 'uci'), "Move should be a Move object or UCI string"
    
    print("✓ Data generation test passed!")
    return True


def test_data_save_load():
    """Test that data can be saved and loaded."""
    print("\nTesting data save/load...")
    
    # Generate small dataset
    board_states, moves, outcomes = generate_training_data(num_games=2, max_moves=10)
    
    # Save
    test_filename = "test_data.npz"
    save_training_data(board_states, moves, outcomes, test_filename)
    
    # Load
    loaded_states, loaded_moves, loaded_outcomes = load_training_data(test_filename)
    
    # Verify loaded data
    assert len(loaded_states) == len(board_states), "Loaded states should match saved"
    assert len(loaded_moves) == len(moves), "Loaded moves should match saved"
    assert len(loaded_outcomes) == len(outcomes), "Loaded outcomes should match saved"
    
    # Verify shapes
    assert loaded_states.shape == (len(board_states), 8, 8, 12), "Loaded states shape incorrect"
    
    # Verify outcomes match
    np.testing.assert_array_almost_equal(loaded_outcomes, np.array(outcomes), 
                                         err_msg="Loaded outcomes should match saved")
    
    print("✓ Data save/load test passed!")
    
    # Cleanup
    import os
    if os.path.exists(test_filename):
        os.remove(test_filename)
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Testing: Training Data Generation")
    print("=" * 60)
    
    try:
        test_data_generation()
        test_data_save_load()
        
        print("\n" + "=" * 60)
        print("✓ All Phase 2 tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

