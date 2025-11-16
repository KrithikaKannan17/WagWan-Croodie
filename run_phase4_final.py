"""
Phase 4: Final Training on ALL Accumulated Data
Uses Modal GPU for fast training on combined dataset
"""

import subprocess
import numpy as np
import sys
import os

def merge_all_data():
    """Merge all available training data."""
    print("\n" + "="*70)
    print("📊 MERGING ALL TRAINING DATA")
    print("="*70 + "\n")
    
    data_files = [
        "tactical_data_all_merged.npz",  # 8,150 tactical positions
        "datasets/selfplay_v1.npz",       # 1,970 self-play positions
        "selfplay_v3_fixed.npz",          # 6,195 self-play positions
    ]
    
    all_boards = []
    all_policies = []
    all_values = []
    
    for f in data_files:
        try:
            if not os.path.exists(f):
                print(f"⚠️  Skipping {f} (not found)")
                continue
            
            data = np.load(f)
            
            # Handle both old and new formats
            if 'boards' in data:
                all_boards.append(data['boards'])
                all_policies.append(data['policy_targets'])
                all_values.append(data['value_targets'])
                print(f"✓ Loaded {f}: {len(data['boards'])} positions")
            else:
                print(f"⚠️  Skipping {f} (incompatible format)")
        except Exception as e:
            print(f"✗ Failed to load {f}: {e}")
    
    if not all_boards:
        print("\n❌ No data files found!")
        return None
    
    # Concatenate
    boards = np.concatenate(all_boards)
    policies = np.concatenate(all_policies)
    values = np.concatenate(all_values)
    
    # Shuffle
    indices = np.random.permutation(len(boards))
    boards = boards[indices]
    policies = policies[indices]
    values = values[indices]
    
    output = 'phase4_final_training_data.npz'
    np.savez_compressed(output,
        boards=boards,
        policy_targets=policies,
        value_targets=values
    )
    
    print(f"\n✅ Merged {len(boards)} total positions → {output}")
    print(f"   Boards: {boards.shape}")
    print(f"   Policies: {policies.shape}")
    print(f"   Values: {values.shape}")
    
    return output


def upload_to_modal(data_file):
    """Upload data to Modal volume."""
    print("\n" + "="*70)
    print("📤 UPLOADING TO MODAL")
    print("="*70 + "\n")
    
    cmd = [
        "modal", "volume", "put",
        "chess-training-data",
        data_file,
        f"/data/data/{data_file}"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Upload failed")
        return False
    
    print("✅ Upload successful")
    return True


def train_on_modal(data_file, model_input, model_output, epochs=30):
    """Train on Modal GPU."""
    print("\n" + "="*70)
    print(f"🚀 TRAINING ON MODAL GPU ({epochs} epochs)")
    print("="*70 + "\n")
    
    cmd = [
        "modal", "run", "train_modal_selfplay.py",
        "--data-file", data_file,
        "--model-input", model_input,
        "--model-output", model_output,
        "--epochs", str(epochs),
        "--batch-size", "128",
        "--learning-rate", "0.002"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Training failed")
        return False
    
    print("✅ Training successful")
    return True


def download_from_modal(model_output):
    """Download trained model from Modal."""
    print("\n" + "="*70)
    print("📥 DOWNLOADING MODEL")
    print("="*70 + "\n")
    
    cmd = [
        "modal", "volume", "get",
        "chess-training-data",
        f"data/data/{model_output}",
        model_output
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Download failed")
        return False
    
    print(f"✅ Downloaded: {model_output}")
    return True


def evaluate_model(model1, model2, games=20):
    """Evaluate the new model against the old one."""
    print("\n" + "="*70)
    print(f"🎮 EVALUATING: {model2} vs {model1}")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "-m", "src.evaluation.test_model_comparison",
        "--model1", model1,
        "--model2", model2,
        "--games", str(games),
        "--mcts-sims", "40",
        "--skip-policy",
        "--skip-value"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║               🚀 PHASE 4: FINAL MODEL TRAINING 🚀                ║
║                                                                  ║
║  1. Merge ALL training data (tactical + self-play)              ║
║  2. Upload to Modal GPU                                          ║
║  3. Train for 30 epochs with optimal hyperparameters            ║
║  4. Download final model                                         ║
║  5. Evaluate against baseline                                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    model_input = "chess_model_sp_v3.pth"  # Best model so far
    model_output = "chess_model_FINAL.pth"
    epochs = 30
    
    if not os.path.exists(model_input):
        print(f"❌ Input model not found: {model_input}")
        print("   Please train Phase 2-3 first, or use chess_model_merged_v1.pth")
        return 1
    
    # Step 1: Merge data
    data_file = merge_all_data()
    if not data_file:
        return 1
    
    # Step 2: Upload to Modal
    if not upload_to_modal(data_file):
        return 1
    
    # Step 3: Train on Modal
    if not train_on_modal(data_file, model_input, model_output, epochs):
        return 1
    
    # Step 4: Download model
    if not download_from_modal(model_output):
        return 1
    
    # Step 5: Evaluate
    evaluate_model(model_input, model_output, games=20)
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    ✅ PHASE 4 COMPLETE! ✅                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n🎉 Final model ready: {model_output}")
    print(f"\n📋 Next steps:")
    print(f"  1. Copy to deployment:")
    print(f"     cp {model_output} chess_model.pth")
    print(f"  2. Push to SECOND_VERSION branch")
    print(f"  3. Deploy to ChessHacks platform")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

