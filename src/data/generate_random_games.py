"""
Generate random chess games for initial training.
This creates diverse data to bootstrap the model.
"""

import chess
import numpy as np
import random
from src.utils.board_encoder import board_to_tensor
from src.utils.move_mapper import MoveMapper


def play_random_game(move_mapper, max_moves=80):
    """
    Play a random game with some basic heuristics.

    Returns:
        positions, policies (uniform), values
    """
    board = chess.Board()
    positions = []
    policies = []

    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            break

        # Encode position
        board_tensor = board_to_tensor(board)
        positions.append(board_tensor)

        # Create uniform policy over legal moves
        policy = np.zeros(256)

        for move in legal_moves:
            move_idx = move_mapper.get_move_index(move)
            if move_idx is not None and move_idx < 256:
                policy[move_idx] = 1.0

        # Normalize
        if policy.sum() > 0:
            policy = policy / policy.sum()

        policies.append(policy)

        # Choose random move (with slight preference for captures)
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures and random.random() < 0.3:
            move = random.choice(captures)
        else:
            move = random.choice(legal_moves)

        board.push(move)
        move_count += 1

    # Determine outcome
    if board.is_checkmate():
        outcome = 1 if board.turn == chess.BLACK else -1
    else:
        outcome = 0

    # Create value targets (from perspective of player to move)
    values = []
    for i in range(len(positions)):
        player = 1 if i % 2 == 0 else -1
        values.append(outcome * player)

    return positions, policies, values


def generate_random_dataset(num_games=1000, output_path="random_games.npz"):
    """Generate dataset of random games."""
    print("="*70)
    print("🎲 Generating Random Chess Games")
    print("="*70)
    print(f"Games: {num_games}")
    print(f"Output: {output_path}")
    print("="*70)

    # Create shared move mapper
    move_mapper = MoveMapper()

    all_positions = []
    all_policies = []
    all_values = []

    wins = 0
    draws = 0
    losses = 0

    for i in range(num_games):
        positions, policies, values = play_random_game(move_mapper)
        
        all_positions.extend(positions)
        all_policies.extend(policies)
        all_values.extend(values)
        
        # Track outcomes
        if values[0] > 0:
            wins += 1
        elif values[0] < 0:
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 100 == 0:
            print(f"Game {i+1}/{num_games} | Total positions: {len(all_positions)}")
    
    # Convert to numpy arrays
    boards = np.array(all_positions, dtype=np.float32)
    policies_arr = np.array(all_policies, dtype=np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    
    # Save
    np.savez_compressed(
        output_path,
        boards=boards,
        policies=policies_arr,
        values=values_arr
    )
    
    print("="*70)
    print(f"✓ Generated {len(all_positions)} positions from {num_games} games")
    print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"  Saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random chess games")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--output", type=str, default="datasets/random_games.npz", help="Output file")
    
    args = parser.parse_args()
    
    generate_random_dataset(args.games, args.output)

