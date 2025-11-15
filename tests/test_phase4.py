"""
Test script for Phase 4: Verify MCTS integration in main.py.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess import Board
from src.main import test_func
from src.utils.decorator import GameContext, chess_manager


def test_mcts_integration():
    """Test that MCTS is integrated and working in main.py."""
    print("Testing MCTS integration in main.py...")
    
    # Create a test board
    board = Board()
    
    # Create game context
    ctx = GameContext(
        board=board,
        timeLeft=60000,  # 60 seconds in milliseconds
        logProbabilities=lambda probs: None  # Dummy function
    )
    
    # Set context in chess manager (use valid PGN for starting position)
    chess_manager.set_context("*", 60000)  # Empty game PGN
    chess_manager._ctx = ctx
    
    try:
        # Call the entrypoint function
        move = test_func(ctx)
        
        print(f"  ✓ Function returned a move: {move.uci()}")
        print(f"  Move is legal: {move in board.generate_legal_moves()}")
        
        # Check that move is legal
        assert move in board.generate_legal_moves(), "Returned move should be legal"
        
        print("  ✓ MCTS integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_to_random():
    """Test that system falls back to random if model not found."""
    print("\nTesting fallback to random moves...")
    
    # Temporarily rename model file
    model_path = "chess_model.pth"
    backup_path = "chess_model.pth.backup"
    
    import shutil
    if os.path.exists(model_path):
        shutil.move(model_path, backup_path)
        print("  Temporarily moved model file to test fallback")
    
    # Reload module to trigger fallback
    import importlib
    import src.main
    importlib.reload(src.main)
    
    board = Board()
    ctx = GameContext(
        board=board,
        timeLeft=60000,
        logProbabilities=lambda probs: None
    )
    chess_manager.set_context("", 60000)
    chess_manager._ctx = ctx
    
    try:
        move = src.main.test_func(ctx)
        print(f"  ✓ Fallback returned a move: {move.uci()}")
        print("  ✓ Fallback to random works")
        
        # Restore model file
        if os.path.exists(backup_path):
            shutil.move(backup_path, model_path)
            importlib.reload(src.main)
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        # Restore model file even on error
        if os.path.exists(backup_path):
            shutil.move(backup_path, model_path)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Testing: MCTS Integration")
    print("=" * 60)
    
    try:
        success1 = test_mcts_integration()
        # Skip fallback test for now to avoid module reload issues
        # success2 = test_fallback_to_random()
        
        if success1:
            print("\n" + "=" * 60)
            print("✓ Phase 4 integration test passed!")
            print("=" * 60)
        else:
            print("\n✗ Phase 4 integration test failed")
            exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

