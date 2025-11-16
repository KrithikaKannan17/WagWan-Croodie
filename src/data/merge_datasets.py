"""
Merge multiple training datasets into one unified dataset.
Handles both old format (board_states, moves, outcomes) and new format (boards, policy_targets, value_targets).
"""

import argparse
import numpy as np
from src.utils.move_mapper import MoveMapper


def convert_old_to_new_format(board_states, moves, outcomes, move_mapper):
    """
    Convert old training data format to new self-play format.
    
    Args:
        board_states: (N, 8, 8, 12) or (N, 12, 8, 8)
        moves: List of UCI move strings
        outcomes: (N,) game outcomes
        move_mapper: MoveMapper instance
    
    Returns:
        Tuple of (boards, policy_targets, value_targets)
    """
    N = len(board_states)
    
    # Convert boards to (N, 12, 8, 8) if needed
    if board_states.shape[1] == 8:
        boards = np.transpose(board_states, (0, 3, 1, 2))
    else:
        boards = board_states
    
    # Create policy targets (one-hot encoding of moves)
    policy_targets = np.zeros((N, 256), dtype=np.float32)
    for i, move in enumerate(moves):
        move_idx = move_mapper.get_move_index(move)
        if move_idx < 256:
            policy_targets[i, move_idx] = 1.0
    
    # Value targets are already in correct format
    value_targets = outcomes.astype(np.float32)
    
    return boards, policy_targets, value_targets


def load_dataset(filepath, move_mapper):
    """
    Load a dataset and convert to unified format.
    
    Args:
        filepath: Path to .npz file
        move_mapper: MoveMapper instance
    
    Returns:
        Tuple of (boards, policy_targets, value_targets)
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Check if it's new format (self-play) or old format
    if 'boards' in data and 'policy_targets' in data:
        # New format
        boards = data['boards']
        policy_targets = data['policy_targets']
        value_targets = data['value_targets']
    elif 'board_states' in data and 'moves' in data:
        # Old format - convert
        print(f"  Converting old format to new format...")
        boards, policy_targets, value_targets = convert_old_to_new_format(
            data['board_states'],
            data['moves'],
            data['outcomes'],
            move_mapper
        )
    else:
        raise ValueError(f"Unknown dataset format in {filepath}")
    
    return boards, policy_targets, value_targets


def main():
    parser = argparse.ArgumentParser(description='Merge training datasets')
    parser.add_argument('--datasets', nargs='+',
                       default=['datasets/selfplay_data.npz', 'datasets/training_data_20k.npz'],
                       help='List of dataset files to merge')
    parser.add_argument('--output', type=str, default='datasets/training_data_merged.npz',
                       help='Output merged dataset file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("📦 Dataset Merger")
    print("="*70)
    print(f"Datasets to merge: {len(args.datasets)}")
    for ds in args.datasets:
        print(f"  - {ds}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Create move mapper
    move_mapper = MoveMapper()
    
    all_boards = []
    all_policies = []
    all_values = []
    
    # Load and merge all datasets
    for dataset_path in args.datasets:
        try:
            print(f"\nLoading {dataset_path}...")
            boards, policies, values = load_dataset(dataset_path, move_mapper)
            
            print(f"  ✓ Loaded {len(boards)} positions")
            print(f"    Board shape: {boards.shape}")
            print(f"    Policy shape: {policies.shape}")
            print(f"    Value shape: {values.shape}")
            
            all_boards.append(boards)
            all_policies.append(policies)
            all_values.append(values)
            
        except FileNotFoundError:
            print(f"  ⚠ File not found: {dataset_path}, skipping...")
        except Exception as e:
            print(f"  ✗ Error loading {dataset_path}: {e}")
    
    if not all_boards:
        print("\n✗ No datasets loaded successfully!")
        return
    
    # Concatenate all datasets
    print(f"\n{'='*70}")
    print("Merging datasets...")
    
    merged_boards = np.concatenate(all_boards, axis=0)
    merged_policies = np.concatenate(all_policies, axis=0)
    merged_values = np.concatenate(all_values, axis=0)
    
    print(f"✓ Merged dataset:")
    print(f"  Total positions: {len(merged_boards)}")
    print(f"  Board shape: {merged_boards.shape}")
    print(f"  Policy shape: {merged_policies.shape}")
    print(f"  Value shape: {merged_values.shape}")
    
    # Shuffle the merged dataset
    print("\nShuffling merged dataset...")
    indices = np.random.permutation(len(merged_boards))
    merged_boards = merged_boards[indices]
    merged_policies = merged_policies[indices]
    merged_values = merged_values[indices]
    
    # Save merged dataset
    print(f"\nSaving to {args.output}...")
    np.savez_compressed(
        args.output,
        boards=merged_boards,
        policy_targets=merged_policies,
        value_targets=merged_values
    )
    
    print(f"\n{'='*70}")
    print(f"✓ Merged dataset saved to {args.output}")
    print(f"  Total positions: {len(merged_boards)}")
    print(f"  Wins: {np.sum(merged_values > 0)}")
    print(f"  Draws: {np.sum(merged_values == 0)}")
    print(f"  Losses: {np.sum(merged_values < 0)}")
    print(f"\nNext step:")
    print(f"  python -m src.training.train_selfplay --data {args.output} --epochs 50")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

