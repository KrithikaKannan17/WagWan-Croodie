"""
Merge selfplay + tactical data and upload for training
"""

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tactical-ratio', type=float, default=0.3, 
                       help='Ratio of tactical data (0-1)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔗 MERGING SELFPLAY + TACTICAL DATA")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    sp1 = np.load('datasets/selfplay_v1.npz')
    sp2 = np.load('selfplay_data.npz')
    tactical = np.load('tactical_data_all_merged.npz')
    
    print(f"  Self-play v1: {len(sp1['boards'])} positions")
    print(f"  Self-play v2: {len(sp2['boards'])} positions")
    print(f"  Tactical: {len(tactical['boards'])} positions")
    
    # Combine selfplay
    sp_boards = np.concatenate([sp1['boards'], sp2['boards']])
    sp_policy = np.concatenate([sp1['policy_targets'], sp2['policy_targets']])
    sp_values = np.concatenate([sp1['value_targets'], sp2['value_targets']])
    
    total_selfplay = len(sp_boards)
    total_tactical = len(tactical['boards'])
    
    print(f"\nTotal self-play: {total_selfplay}")
    print(f"Total tactical: {total_tactical}")
    
    # Calculate how many tactical samples to include
    tactical_ratio = args.tactical_ratio
    selfplay_ratio = 1.0 - tactical_ratio
    
    # Target: make tactical_count / (tactical_count + selfplay_count) = tactical_ratio
    # If we keep all tactical and sample selfplay:
    #   total_tactical / (total_tactical + sampled_selfplay) = tactical_ratio
    #   sampled_selfplay = total_tactical * (1 - tactical_ratio) / tactical_ratio
    
    sampled_selfplay_count = int(total_tactical * (1 - tactical_ratio) / tactical_ratio)
    
    if sampled_selfplay_count > total_selfplay:
        # If we need more selfplay than we have, keep all selfplay and sample tactical
        sampled_selfplay_count = total_selfplay
        sampled_tactical_count = int(total_selfplay * tactical_ratio / (1 - tactical_ratio))
    else:
        sampled_tactical_count = total_tactical
    
    print(f"\n📊 Target ratio: {tactical_ratio*100:.0f}% tactical, {selfplay_ratio*100:.0f}% self-play")
    print(f"   Using {sampled_tactical_count} tactical + {sampled_selfplay_count} self-play")
    
    # Sample
    tactical_indices = np.random.choice(total_tactical, sampled_tactical_count, replace=False)
    selfplay_indices = np.random.choice(total_selfplay, sampled_selfplay_count, replace=False)
    
    # Combine
    merged_boards = np.concatenate([
        tactical['boards'][tactical_indices],
        sp_boards[selfplay_indices]
    ])
    merged_policy = np.concatenate([
        tactical['policy_targets'][tactical_indices],
        sp_policy[selfplay_indices]
    ])
    merged_values = np.concatenate([
        tactical['value_targets'][tactical_indices],
        sp_values[selfplay_indices]
    ])
    
    # Shuffle
    print("\nShuffling...")
    indices = np.random.permutation(len(merged_boards))
    merged_boards = merged_boards[indices]
    merged_policy = merged_policy[indices]
    merged_values = merged_values[indices]
    
    # Save
    output_file = 'merged_tactical_selfplay.npz'
    np.savez_compressed(
        output_file,
        boards=merged_boards,
        policy_targets=merged_policy,
        value_targets=merged_values
    )
    
    print("\n" + "="*70)
    print("✅ MERGED DATASET SAVED")
    print("="*70)
    print(f"  File: {output_file}")
    print(f"  Total positions: {len(merged_boards)}")
    print(f"  Actual tactical ratio: {sampled_tactical_count / len(merged_boards) * 100:.1f}%")
    print(f"  Boards shape: {merged_boards.shape}")
    print("="*70)
    
    print("\n🎯 Next steps:")
    print(f"  1. Upload to Modal:")
    print(f"     modal volume put chess-training-data {output_file} /data/{output_file}")
    print(f"\n  2. Train on Modal:")
    print(f"     modal run train_modal_selfplay.py \\")
    print(f"       --data-file {output_file} \\")
    print(f"       --model-input chess_model_sp_v2.pth \\")
    print(f"       --model-output chess_model_merged_v1.pth \\")
    print(f"       --epochs 10")
    print(f"\n  3. Download and evaluate:")
    print(f"     modal run train_modal_selfplay.py --model-output chess_model_merged_v1.pth --download")
    print(f"     python -m src.evaluation.test_model_comparison \\")
    print(f"       --model1 chess_model_sp_v2.pth \\")
    print(f"       --model2 chess_model_merged_v1.pth \\")
    print(f"       --games 20 --skip-policy --skip-value")


if __name__ == "__main__":
    main()

