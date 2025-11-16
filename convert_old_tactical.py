"""
Convert old tactical data format to new training format
Old: board_states (N, 8, 8, 12), moves (N,), outcomes (N,)
New: boards (N, 12, 8, 8), policy_targets (N, 256), value_targets (N,)
"""

import numpy as np
from src.utils.move_mapper import MoveMapper

def convert_old_tactical_file(old_file: str, output_file: str):
    """Convert old tactical format to new format."""
    print(f"\n🔄 Converting {old_file}...")
    
    # Load old format
    data = np.load(old_file)
    board_states = data['board_states']  # (N, 8, 8, 12)
    moves = data['moves']  # (N,) - UCI strings
    outcomes = data['outcomes']  # (N,) - values
    
    N = len(board_states)
    print(f"  Loaded {N} positions")
    
    # Convert boards: (N, 8, 8, 12) -> (N, 12, 8, 8)
    boards = np.transpose(board_states, (0, 3, 1, 2))
    
    # Convert moves to policy targets
    move_mapper = MoveMapper()
    policy_targets = np.zeros((N, 256), dtype=np.float32)
    
    valid_count = 0
    for i in range(N):
        try:
            move_str = str(moves[i])
            move_idx = move_mapper.get_move_index_from_uci(move_str)
            policy_targets[i, move_idx] = 1.0
            valid_count += 1
        except Exception as e:
            # If move mapping fails, create a uniform distribution
            policy_targets[i] = 1.0 / 256
    
    print(f"  Successfully mapped {valid_count}/{N} moves")
    
    # Value targets (outcomes are already in correct format)
    value_targets = outcomes.astype(np.float32)
    
    # Save in new format
    np.savez_compressed(
        output_file,
        boards=boards.astype(np.float32),
        policy_targets=policy_targets,
        value_targets=value_targets
    )
    
    print(f"  ✅ Saved to {output_file}")
    print(f"     Shape: boards={boards.shape}, policy={policy_targets.shape}, value={value_targets.shape}")
    return N


def main():
    print("=" * 70)
    print("🔧 CONVERTING OLD TACTICAL DATA TO NEW FORMAT")
    print("=" * 70)
    
    files_to_convert = [
        ('fork_test.npz', 'fork_converted.npz'),
        ('hanging_test.npz', 'hanging_converted.npz'),
        ('pin_test.npz', 'pin_converted.npz'),
        ('tactical_data.npz', 'tactical_old_converted.npz'),
        ('test_tactical.npz', 'test_tactical_converted.npz'),
    ]
    
    total_positions = 0
    converted_files = []
    
    for old_file, new_file in files_to_convert:
        try:
            count = convert_old_tactical_file(old_file, new_file)
            total_positions += count
            converted_files.append(new_file)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ CONVERSION COMPLETE: {total_positions} positions")
    print("=" * 70)
    
    # Merge all tactical data
    print("\n🔗 MERGING ALL TACTICAL DATA...")
    
    all_boards = []
    all_policy = []
    all_values = []
    
    # Include the new tactical_data_phase3.npz
    for npz_file in converted_files + ['tactical_data_phase3.npz']:
        try:
            data = np.load(npz_file)
            all_boards.append(data['boards'])
            all_policy.append(data['policy_targets'])
            all_values.append(data['value_targets'])
            print(f"  ✓ Loaded {npz_file}: {len(data['boards'])} positions")
        except Exception as e:
            print(f"  ✗ Failed to load {npz_file}: {e}")
    
    # Concatenate
    merged_boards = np.concatenate(all_boards)
    merged_policy = np.concatenate(all_policy)
    merged_values = np.concatenate(all_values)
    
    # Shuffle
    indices = np.random.permutation(len(merged_boards))
    merged_boards = merged_boards[indices]
    merged_policy = merged_policy[indices]
    merged_values = merged_values[indices]
    
    # Save
    output_file = 'tactical_data_all_merged.npz'
    np.savez_compressed(
        output_file,
        boards=merged_boards,
        policy_targets=merged_policy,
        value_targets=merged_values
    )
    
    print("\n" + "=" * 70)
    print(f"✅ MERGED TACTICAL DATASET SAVED")
    print("=" * 70)
    print(f"  File: {output_file}")
    print(f"  Total positions: {len(merged_boards)}")
    print(f"  Boards shape: {merged_boards.shape}")
    print(f"  Policy targets shape: {merged_policy.shape}")
    print(f"  Value targets shape: {merged_values.shape}")
    print("=" * 70)
    
    print("\n🎯 Next step:")
    print(f"  modal volume put chess-training-data {output_file} /data/{output_file}")


if __name__ == "__main__":
    main()

