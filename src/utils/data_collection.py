"""
Data collection script for generating training data from self-play games.
Generates games using random moves and records board states with moves.
"""

import random
from chess import Board, Move
from typing import List, Tuple
import numpy as np
from .board_encoder import board_to_tensor


def generate_random_game(max_moves: int = 200) -> Tuple[List[np.ndarray], List[Move], List[bool], int]:
    """
    Generate a single game using random legal moves.
    
    Args:
        max_moves: Maximum number of moves before stopping the game
        
    Returns:
        board_states: List of board state tensors (8x8x12)
        moves: List of moves made
        is_white_turn: List of booleans indicating if it was white's turn
        result: Game result (1 for white win, -1 for black win, 0 for draw)
    """
    board = Board()
    board_states = []
    moves = []
    is_white_turn = []
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        # Record current board state and whose turn it is
        board_state = board_to_tensor(board)
        board_states.append(board_state)
        is_white_turn.append(board.turn)  # True if white's turn
        
        # Get legal moves
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            break
        
        # Choose random move
        move = random.choice(legal_moves)
        moves.append(move)
        board.push(move)
        move_count += 1
    
    # Determine game result
    if board.is_checkmate():
        # If it's white's turn now, black just moved and won
        result = -1 if board.turn else 1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = 0  # Draw
    else:
        result = 0  # Draw (timeout or other)
    
    return board_states, moves, is_white_turn, result


def generate_training_data(num_games: int = 100, max_moves: int = 200) -> Tuple[List[np.ndarray], List[Move], List[float]]:
    """
    Generate training data from multiple random games.
    
    Args:
        num_games: Number of games to generate
        max_moves: Maximum moves per game
        
    Returns:
        board_states: List of all board state tensors
        moves: List of all moves made
        outcomes: List of game outcomes (1, -1, or 0) for each position
    """
    all_board_states = []
    all_moves = []
    all_outcomes = []
    
    print(f"Generating {num_games} games...")
    for game_num in range(num_games):
        if (game_num + 1) % 10 == 0:
            print(f"  Generated {game_num + 1}/{num_games} games...")
        
        board_states, moves, is_white_turn, result = generate_random_game(max_moves)
        
        # For each position, assign the outcome from the perspective of the player to move
        for board_state, move, white_to_move in zip(board_states, moves, is_white_turn):
            all_board_states.append(board_state)
            all_moves.append(move)
            
            # Determine outcome from perspective of player who made the move
            if result == 0:
                outcome = 0.0  # Draw
            elif white_to_move:
                # White's turn - outcome is from white's perspective
                outcome = float(result)  # 1 if white won, -1 if black won
            else:
                # Black's turn - outcome is from black's perspective
                outcome = float(-result)  # -1 if white won, 1 if black won
            
            all_outcomes.append(outcome)
    
    print(f"Generated {len(all_board_states)} training positions from {num_games} games")
    return all_board_states, all_moves, all_outcomes


def save_training_data(board_states: List[np.ndarray], moves: List[Move], 
                      outcomes: List[float], filename: str = "training_data.npz"):
    """
    Save training data to a numpy compressed file.
    
    Args:
        board_states: List of board state tensors
        moves: List of moves
        outcomes: List of outcomes
        filename: Output filename
    """
    # Convert board states to numpy array
    board_array = np.array(board_states)  # Shape: (N, 8, 8, 12)
    
    # Convert moves to indices (we'll need a move-to-index mapping)
    # For now, save moves as UCI strings
    move_strings = [move.uci() for move in moves]
    
    # Convert outcomes to numpy array
    outcomes_array = np.array(outcomes, dtype=np.float32)
    
    # Save to file
    np.savez_compressed(
        filename,
        board_states=board_array,
        moves=move_strings,
        outcomes=outcomes_array
    )
    
    print(f"Saved {len(board_states)} training positions to {filename}")


def load_training_data(filename: str = "training_data.npz") -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load training data from a numpy compressed file.
    
    Args:
        filename: Input filename
        
    Returns:
        board_states: Array of board states (N, 8, 8, 12)
        moves: List of move UCI strings
        outcomes: Array of outcomes
    """
    data = np.load(filename, allow_pickle=True)
    board_states = data['board_states']
    moves = data['moves'].tolist()
    outcomes = data['outcomes']
    
    return board_states, moves, outcomes

