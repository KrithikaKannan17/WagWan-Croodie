"""
Pipeline.py - End-to-End Training Orchestration
Coordinates all phases of training from baseline to final model
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse


class TrainingPipeline:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self, use_modal=True, verbose=True):
        self.use_modal = use_modal
        self.verbose = verbose
        self.models = {}
        self.data_files = {}
        
    def log(self, message):
        """Print log message if verbose."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📋 {message}")
            print(f"{'='*70}\n")
    
    def run_command(self, cmd, description):
        """Run a command and handle errors."""
        self.log(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"❌ Failed: {description}")
            return False
        print(f"✅ Success: {description}")
        return True
    
    def run_phase1_baseline(self):
        """Phase 1: Use pre-trained baseline model."""
        self.log("PHASE 1: Baseline Model Setup")
        
        # Check if baseline model exists
        if os.path.exists("chess_model_best.pth"):
            print("✅ Found baseline model: chess_model_best.pth")
            self.models['phase1_baseline'] = "chess_model_best.pth"
            return True
        else:
            print("❌ Baseline model not found: chess_model_best.pth")
            print("   Please download from Lichess or training source")
            return False
    
    def run_phase2_selfplay_bootstrap(self, model_path, iterations=3, games_per_iter=200):
        """Phase 2: Self-play iterations to bootstrap the model."""
        self.log(f"PHASE 2: Self-Play Bootstrap ({iterations} iterations)")
        
        current_model = model_path
        
        for iteration in range(1, iterations + 1):
            print(f"\n{'#'*70}")
            print(f"# ITERATION {iteration}/{iterations}")
            print(f"# Current model: {current_model}")
            print(f"{'#'*70}\n")
            
            # Generate self-play data
            selfplay_file = f"selfplay_iter{iteration}.npz"
            if self.use_modal:
                if not self.run_command([
                    "modal", "run", "selfplay_modal.py",
                    "--games", str(games_per_iter),
                    "--model-path", current_model,
                    "--output-name", selfplay_file,
                    "--mcts-sims", "40"
                ], f"Generate {games_per_iter} self-play games (Modal GPU)"):
                    return False
            else:
                if not self.run_command([
                    sys.executable, "selfplay_generator.py",
                    "--games", str(games_per_iter),
                    "--model", current_model,
                    "--output", f"./{selfplay_file}",
                    "--mcts-sims", "40"
                ], f"Generate {games_per_iter} self-play games (Local)"):
                    return False
            
            self.data_files[f'selfplay_iter{iteration}'] = selfplay_file
            
            # Train on self-play data
            next_model = f"chess_model_sp_v{iteration}.pth"
            if self.use_modal:
                if not self.run_command([
                    "modal", "run", "train_modal_selfplay.py",
                    "--data-file", selfplay_file,
                    "--model-input", current_model,
                    "--model-output", next_model,
                    "--epochs", "8",
                    "--batch-size", "128",
                    "--learning-rate", "0.001"
                ], f"Train iteration {iteration} (Modal GPU)"):
                    return False
                
                # Download model
                if not self.run_command([
                    "modal", "volume", "get",
                    "chess-training-data",
                    f"data/data/{next_model}",
                    next_model
                ], f"Download {next_model}"):
                    return False
            else:
                if not self.run_command([
                    sys.executable, "-m", "src.training.train_selfplay",
                    "--data", f"./{selfplay_file}",
                    "--model", current_model,
                    "--output", next_model,
                    "--epochs", "8"
                ], f"Train iteration {iteration} (Local)"):
                    return False
            
            current_model = next_model
            self.models[f'phase2_iter{iteration}'] = next_model
        
        self.models['phase2_final'] = current_model
        return True
    
    def run_phase3_tactics(self, model_path, num_tactical_positions=5000):
        """Phase 3: Generate tactical data and blend with self-play."""
        self.log("PHASE 3: Tactical Position Training")
        
        # Generate tactical data
        tactical_file = "tactical_phase3.npz"
        if not self.run_command([
            sys.executable, "tactical_generator.py",
            "--num-positions", str(num_tactical_positions),
            "--output", tactical_file
        ], f"Generate {num_tactical_positions} tactical positions"):
            return False
        
        self.data_files['tactical'] = tactical_file
        
        # Merge tactical + self-play data
        print("\n📊 Merging tactical + self-play data...")
        # Use existing selfplay data from Phase 2
        
        # Train on blended data
        blended_model = "chess_model_phase3_blend.pth"
        if self.use_modal:
            # Upload tactical data
            if not self.run_command([
                "modal", "volume", "put",
                "chess-training-data",
                tactical_file,
                f"/data/data/{tactical_file}"
            ], "Upload tactical data to Modal"):
                return False
            
            # Train on Modal
            if not self.run_command([
                "modal", "run", "train_modal_selfplay.py",
                "--data-file", tactical_file,
                "--model-input", model_path,
                "--model-output", blended_model,
                "--epochs", "15",
                "--learning-rate", "0.001"
            ], "Train on tactical data (Modal GPU)"):
                return False
            
            # Download
            if not self.run_command([
                "modal", "volume", "get",
                "chess-training-data",
                f"data/data/{blended_model}",
                blended_model
            ], f"Download {blended_model}"):
                return False
        else:
            if not self.run_command([
                sys.executable, "-m", "src.training.train_selfplay",
                "--data", f"./{tactical_file}",
                "--model", model_path,
                "--output", blended_model,
                "--epochs", "15"
            ], "Train on tactical data (Local)"):
                return False
        
        self.models['phase3_blend'] = blended_model
        return True
    
    def run_phase4_final_training(self, model_path):
        """Phase 4: Final training with all accumulated data on Modal GPU."""
        self.log("PHASE 4: Final Modal GPU Training")
        
        # Merge ALL data files
        print("\n📊 Merging ALL training data...")
        all_data_files = list(self.data_files.values())
        print(f"   Data files to merge: {all_data_files}")
        
        # Create merged dataset
        merge_script = """
import numpy as np
import sys

files = sys.argv[1:]
output = 'final_training_data_merged.npz'

all_boards = []
all_policies = []
all_values = []

for f in files:
    try:
        data = np.load(f)
        all_boards.append(data['boards'])
        all_policies.append(data['policy_targets'])
        all_values.append(data['value_targets'])
        print(f'✓ Loaded {f}: {len(data[\"boards\"])} positions')
    except Exception as e:
        print(f'✗ Failed to load {f}: {e}')

boards = np.concatenate(all_boards)
policies = np.concatenate(all_policies)
values = np.concatenate(all_values)

# Shuffle
indices = np.random.permutation(len(boards))
boards = boards[indices]
policies = policies[indices]
values = values[indices]

np.savez_compressed(output,
    boards=boards,
    policy_targets=policies,
    value_targets=values
)

print(f'\\n✅ Merged {len(boards)} positions → {output}')
"""
        
        # Write and run merge script
        with open('_merge_data.py', 'w') as f:
            f.write(merge_script)
        
        cmd = [sys.executable, '_merge_data.py'] + all_data_files
        if not self.run_command(cmd, "Merge all training data"):
            return False
        
        merged_data = "final_training_data_merged.npz"
        
        # Upload to Modal and train
        final_model = "chess_model_final.pth"
        if self.use_modal:
            if not self.run_command([
                "modal", "volume", "put",
                "chess-training-data",
                merged_data,
                f"/data/data/{merged_data}"
            ], "Upload merged data to Modal"):
                return False
            
            if not self.run_command([
                "modal", "run", "train_modal_selfplay.py",
                "--data-file", merged_data,
                "--model-input", model_path,
                "--model-output", final_model,
                "--epochs", "30",
                "--batch-size", "128",
                "--learning-rate", "0.002"
            ], "Final training on Modal GPU (30 epochs)"):
                return False
            
            if not self.run_command([
                "modal", "volume", "get",
                "chess-training-data",
                f"data/data/{final_model}",
                final_model
            ], f"Download {final_model}"):
                return False
        else:
            if not self.run_command([
                sys.executable, "-m", "src.training.train_selfplay",
                "--data", f"./{merged_data}",
                "--model", model_path,
                "--output", final_model,
                "--epochs", "30"
            ], "Final training (Local)"):
                return False
        
        self.models['phase4_final'] = final_model
        
        # Cleanup
        if os.path.exists('_merge_data.py'):
            os.remove('_merge_data.py')
        
        return True
    
    def evaluate_final_model(self, model_path):
        """Evaluate the final model."""
        self.log("FINAL EVALUATION")
        
        # Evaluate against baseline
        baseline = self.models.get('phase1_baseline', 'chess_model_best.pth')
        self.run_command([
            sys.executable, "-m", "src.evaluation.test_model_comparison",
            "--model1", baseline,
            "--model2", model_path,
            "--games", "20",
            "--mcts-sims", "40",
            "--skip-policy",
            "--skip-value"
        ], f"Evaluate {model_path} vs baseline")
        
        return True
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                  🚀 FULL TRAINING PIPELINE 🚀                    ║
║                                                                  ║
║  Phase 1: Baseline model setup                                  ║
║  Phase 2: Self-play bootstrap (3 iterations)                    ║
║  Phase 3: Tactical position training                            ║
║  Phase 4: Final GPU training on all data                        ║
║  Final:   Evaluation and model export                           ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        # Phase 1
        if not self.run_phase1_baseline():
            print("❌ Pipeline failed at Phase 1")
            return False
        
        # Phase 2
        if not self.run_phase2_selfplay_bootstrap(
            self.models['phase1_baseline'],
            iterations=3,
            games_per_iter=200
        ):
            print("❌ Pipeline failed at Phase 2")
            return False
        
        # Phase 3
        if not self.run_phase3_tactics(
            self.models['phase2_final'],
            num_tactical_positions=5000
        ):
            print("❌ Pipeline failed at Phase 3")
            return False
        
        # Phase 4
        if not self.run_phase4_final_training(
            self.models['phase3_blend']
        ):
            print("❌ Pipeline failed at Phase 4")
            return False
        
        # Evaluation
        self.evaluate_final_model(self.models['phase4_final'])
        
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                  ✅ PIPELINE COMPLETE! ✅                        ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        print(f"\n📁 Final model: {self.models['phase4_final']}")
        print(f"\n📊 All models generated:")
        for phase, model in self.models.items():
            print(f"   {phase}: {model}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Chess Training Pipeline')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', 'all'],
                       default='all', help='Which phase to run')
    parser.add_argument('--modal', action='store_true', default=True,
                       help='Use Modal GPU (default: True)')
    parser.add_argument('--local', action='store_true',
                       help='Use local CPU (overrides --modal)')
    parser.add_argument('--baseline', type=str, default='chess_model_best.pth',
                       help='Baseline model path')
    
    args = parser.parse_args()
    
    use_modal = args.modal and not args.local
    pipeline = TrainingPipeline(use_modal=use_modal)
    
    if args.phase == 'all':
        success = pipeline.run_full_pipeline()
    elif args.phase == '1':
        success = pipeline.run_phase1_baseline()
    elif args.phase == '2':
        success = pipeline.run_phase2_selfplay_bootstrap(args.baseline, iterations=3)
    elif args.phase == '3':
        success = pipeline.run_phase3_tactics(args.baseline)
    elif args.phase == '4':
        success = pipeline.run_phase4_final_training(args.baseline)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

