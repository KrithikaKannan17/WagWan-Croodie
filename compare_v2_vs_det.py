"""
Compare ULTIMATE_V2 vs DETERMINISTIC.
Test if synthetic data improved the model!
"""

import torch
import chess
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src.utils.model import ChessModel
from src.utils.fixed_move_mapper import FixedMoveMapper
from src.utils.board_encoder import board_to_tensor_torch


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model = ChessModel(num_residual_blocks=6, channels=64, dropout=0.0)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def select_move(model, board, move_mapper):
    board_tensor = board_to_tensor_torch(board)
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
    policy_probs = torch.softmax(policy_logits, dim=1)[0]
    legal_moves = list(board.legal_moves)
    move_probs = [(move, policy_probs[move_mapper.get_move_index(move)].item()) for move in legal_moves]
    move_probs.sort(key=lambda x: x[1], reverse=True)
    return move_probs[0][0] if move_probs else None


def play_game(model1, model2, move_mapper, max_moves=200):
    board = chess.Board()
    moves = 0
    
    while not board.is_game_over() and moves < max_moves:
        if board.turn:
            move = select_move(model1, board, move_mapper)
        else:
            move = select_move(model2, board, move_mapper)
        
        if move is None:
            return 'draw'
        
        board.push(move)
        moves += 1
    
    if board.is_checkmate():
        return 'black' if board.turn else 'white'
    else:
        return 'draw'


def main():
    print("="*70)
    print("🆚 ULTIMATE_V2 vs DETERMINISTIC (20 games)")
    print("="*70)
    print()
    
    print("Loading models...")
    model_v2 = load_model("chess_model_ULTIMATE_V2.pth")
    print("  ✓ ULTIMATE_V2 (56K: GM+Tactics+Checkmates+Defense)")
    print("     Val loss: 4.06")
    
    model_det = load_model("chess_model_DETERMINISTIC.pth")
    print("  ✓ DETERMINISTIC (41K: GM only)")
    print("     Val loss: 4.35")
    
    move_mapper = FixedMoveMapper()
    print()
    
    results = {'v2_wins': 0, 'det_wins': 0, 'draws': 0}
    
    print("Playing 20 games (10 as white, 10 as black)...")
    print()
    
    # 10 games: ULTIMATE_V2 as white
    print("Games 1-10: ULTIMATE_V2 (white) vs DETERMINISTIC (black)")
    print("-"*70)
    for i in range(1, 11):
        result = play_game(model_v2, model_det, move_mapper)
        
        if result == 'white':
            results['v2_wins'] += 1
            outcome = "✅ ULTIMATE_V2 wins"
        elif result == 'black':
            results['det_wins'] += 1
            outcome = "❌ DETERMINISTIC wins"
        else:
            results['draws'] += 1
            outcome = "Draw"
        
        print(f"  Game {i}: {outcome}")
    
    print()
    
    # 10 games: ULTIMATE_V2 as black
    print("Games 11-20: DETERMINISTIC (white) vs ULTIMATE_V2 (black)")
    print("-"*70)
    for i in range(11, 21):
        result = play_game(model_det, model_v2, move_mapper)
        
        if result == 'black':
            results['v2_wins'] += 1
            outcome = "✅ ULTIMATE_V2 wins"
        elif result == 'white':
            results['det_wins'] += 1
            outcome = "❌ DETERMINISTIC wins"
        else:
            results['draws'] += 1
            outcome = "Draw"
        
        print(f"  Game {i}: {outcome}")
    
    print()
    print("="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"ULTIMATE_V2 wins: {results['v2_wins']}/20")
    print(f"DETERMINISTIC wins: {results['det_wins']}/20")
    print(f"Draws: {results['draws']}/20")
    print()
    
    # Analysis
    if results['v2_wins'] > results['det_wins']:
        print("🏆 ULTIMATE_V2 is BETTER!")
        print("   ✅ Synthetic tactics/checkmates/defense training helped!")
    elif results['det_wins'] > results['v2_wins']:
        print("🏆 DETERMINISTIC is BETTER!")
        print("   ✅ Pure GM training is superior!")
    else:
        print("🤝 TIE!")
        print("   Both models are equally strong!")
    
    print()
    print("Validation Loss Comparison:")
    print(f"  ULTIMATE_V2:    4.06 ✅ (lower)")
    print(f"  DETERMINISTIC:  4.35")
    print()
    print("="*70)


if __name__ == "__main__":
    main()

