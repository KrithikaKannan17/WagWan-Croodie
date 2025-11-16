"""
Phase 2: Iterative Self-Play Training Loop
==========================================

Orchestrates the complete AlphaZero-style training cycle:
1. Generate self-play games with current best model
2. Train on self-play data
3. Evaluate new model vs baseline
4. If improved, promote as new baseline
5. Repeat

Usage:
    # Run 3 iterations locally
    python phase2_iterative_loop.py --iterations 3 --games 100 --local

    # Run 3 iterations on Modal GPU
    python phase2_iterative_loop.py --iterations 3 --games 300 --modal

    # Continue from existing model
    python phase2_iterative_loop.py --baseline chess_model_sp_v2.pth --iterations 2
"""

import argparse
import os
import subprocess
import time
from datetime import datetime


class Phase2Trainer:
    def __init__(self, use_modal=False, baseline_model="chess_model_best.pth"):
        self.use_modal = use_modal
        self.baseline_model = baseline_model
        self.iteration = 1
        self.best_model = baseline_model
        self.history = []
        
    def generate_selfplay(self, games=200, mcts_sims=50, output_name=None):
        """Generate self-play games using current best model."""
        if output_name is None:
            output_name = f"datasets/selfplay_v{self.iteration}.npz"
        
        print(f"\n{'='*70}")
        print(f"📊 GENERATING SELF-PLAY DATA - Iteration {self.iteration}")
        print(f"{'='*70}")
        print(f"  Model: {self.best_model}")
        print(f"  Games: {games}")
        print(f"  MCTS sims: {mcts_sims}")
        print(f"  Output: {output_name}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if self.use_modal:
            # Use Modal GPU for generation
            cmd = [
                "modal", "run", "selfplay_modal.py",
                "--games", str(games),
                "--mcts-sims", str(mcts_sims),
                "--model-path", self.best_model,
                "--output-name", os.path.basename(output_name)
            ]
        else:
            # Local generation
            os.makedirs(os.path.dirname(output_name), exist_ok=True)
            cmd = [
                "python", "selfplay_generator.py",
                "--games", str(games),
                "--mcts-sims", str(mcts_sims),
                "--model", self.best_model,
                "--output", output_name
            ]
        
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Self-play generation complete ({elapsed/60:.1f} minutes)")
        return output_name
    
    def train_model(self, data_file, epochs=5, batch_size=128, output_name=None):
        """Train model on self-play data."""
        if output_name is None:
            output_name = f"chess_model_sp_v{self.iteration}.pth"
        
        print(f"\n{'='*70}")
        print(f"🎓 TRAINING MODEL - Iteration {self.iteration}")
        print(f"{'='*70}")
        print(f"  Data: {data_file}")
        print(f"  Base model: {self.best_model}")
        print(f"  Output: {output_name}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if self.use_modal:
            # Use Modal GPU for training
            cmd = [
                "modal", "run", "train_modal_selfplay.py",
                "--data-file", os.path.basename(data_file),
                "--model-input", self.best_model,
                "--model-output", output_name,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size)
            ]
        else:
            # Local training
            cmd = [
                "python", "-m", "src.training.train_selfplay",
                "--data", data_file,
                "--model", self.best_model,
                "--output", output_name,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", "0.001",
                "--patience", "5"
            ]
        
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Training complete ({elapsed/60:.1f} minutes)")
        return output_name
    
    def evaluate_models(self, model1, model2, games=20, mcts_sims=20):
        """Evaluate two models head-to-head."""
        print(f"\n{'='*70}")
        print(f"⚔️  EVALUATING MODELS - Iteration {self.iteration}")
        print(f"{'='*70}")
        print(f"  Baseline: {model1}")
        print(f"  New Model: {model2}")
        print(f"  Games: {games}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        cmd = [
            "python", "-m", "src.evaluation.test_model_comparison",
            "--model1", model1,
            "--model2", model2,
            "--games", str(games),
            "--mcts-sims", str(mcts_sims)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Parse results (look for win rate in output)
        output = result.stdout
        print(output)
        
        # Extract win rate (Model2 win rate)
        win_rate = self._parse_win_rate(output)
        
        print(f"\n✅ Evaluation complete ({elapsed/60:.1f} minutes)")
        print(f"  Model2 Win Rate: {win_rate:.1f}%")
        
        return win_rate
    
    def _parse_win_rate(self, output):
        """Parse win rate from evaluation output."""
        # Look for "Model2 Win Rate: XX.X%"
        import re
        match = re.search(r"Model2 Win Rate: ([\d.]+)%", output)
        if match:
            return float(match.group(1))
        
        # Fallback: count wins manually
        model2_wins = output.count("Model2 wins")
        total_games = output.count("Game ")
        if total_games > 0:
            return (model2_wins / total_games) * 100
        
        return 0.0
    
    def run_iteration(self, games=200, mcts_sims=50, train_epochs=5, eval_games=20):
        """Run one complete iteration of the training loop."""
        print(f"\n{'#'*70}")
        print(f"# ITERATION {self.iteration}")
        print(f"# Baseline: {self.best_model}")
        print(f"{'#'*70}\n")
        
        iteration_start = time.time()
        
        # Step 1: Generate self-play data
        data_file = self.generate_selfplay(games=games, mcts_sims=mcts_sims)
        
        # Step 2: Train model
        new_model = self.train_model(data_file, epochs=train_epochs)
        
        # Step 3: Evaluate
        win_rate = self.evaluate_models(
            self.baseline_model,
            new_model,
            games=eval_games,
            mcts_sims=20
        )
        
        # Step 4: Decide if we promote
        iteration_time = time.time() - iteration_start
        
        result = {
            'iteration': self.iteration,
            'baseline': self.baseline_model,
            'new_model': new_model,
            'data_file': data_file,
            'win_rate': win_rate,
            'time': iteration_time
        }
        
        self.history.append(result)
        
        print(f"\n{'='*70}")
        print(f"📊 ITERATION {self.iteration} SUMMARY")
        print(f"{'='*70}")
        print(f"  Baseline: {self.baseline_model}")
        print(f"  New Model: {new_model}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Time: {iteration_time/60:.1f} minutes")
        
        if win_rate > 55:
            print(f"\n  ✅ NEW MODEL IS STRONGER - Promoting as new baseline!")
            self.best_model = new_model
            self.baseline_model = new_model
        elif win_rate > 45:
            print(f"\n  ➡️  MODELS ARE SIMILAR - Keeping current baseline")
        else:
            print(f"\n  ⚠️  NEW MODEL IS WEAKER - Keeping current baseline")
        
        print(f"{'='*70}\n")
        
        self.iteration += 1
        
        return result
    
    def print_summary(self):
        """Print final summary of all iterations."""
        print(f"\n{'#'*70}")
        print(f"# PHASE 2 TRAINING SUMMARY")
        print(f"{'#'*70}\n")
        
        print(f"Total Iterations: {len(self.history)}")
        print(f"Final Model: {self.best_model}")
        print(f"\nIteration History:")
        print(f"{'='*70}")
        
        for result in self.history:
            status = "✅ PROMOTED" if result['win_rate'] > 55 else "➡️  KEPT" if result['win_rate'] > 45 else "⚠️  REJECTED"
            print(f"  Iteration {result['iteration']}: {result['new_model']}")
            print(f"    Win Rate: {result['win_rate']:.1f}% | {status}")
            print(f"    Time: {result['time']/60:.1f} min")
            print()
        
        print(f"{'='*70}")
        print(f"\n🎯 Ready for Phase 3!")
        print(f"   Use model: {self.best_model}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Iterative Self-Play Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 iterations locally (faster, less data)
  python phase2_iterative_loop.py --iterations 3 --games 100 --local

  # Run 3 iterations on Modal GPU (slower, more data, better quality)
  python phase2_iterative_loop.py --iterations 3 --games 300 --modal

  # Continue from existing model
  python phase2_iterative_loop.py --baseline chess_model_sp_v2.pth --iterations 2

  # Quick test (1 iteration, minimal games)
  python phase2_iterative_loop.py --iterations 1 --games 50 --train-epochs 3 --eval-games 10 --local
        """
    )
    
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of training iterations (default: 3)')
    parser.add_argument('--baseline', type=str, default='chess_model_best.pth',
                       help='Starting baseline model (default: chess_model_best.pth)')
    parser.add_argument('--games', type=int, default=200,
                       help='Self-play games per iteration (default: 200)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS simulations per move (default: 50)')
    parser.add_argument('--train-epochs', type=int, default=5,
                       help='Training epochs per iteration (default: 5)')
    parser.add_argument('--eval-games', type=int, default=20,
                       help='Evaluation games (default: 20)')
    parser.add_argument('--modal', action='store_true',
                       help='Use Modal GPU for generation and training')
    parser.add_argument('--local', action='store_true',
                       help='Use local CPU/GPU (default if neither --modal nor --local specified)')
    
    args = parser.parse_args()
    
    # Default to local if neither specified
    use_modal = args.modal if (args.modal or args.local) else False
    
    print(f"\n{'='*70}")
    print(f"🚀 PHASE 2: ITERATIVE SELF-PLAY TRAINING")
    print(f"{'='*70}")
    print(f"  Mode: {'Modal GPU' if use_modal else 'Local'}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Games/iteration: {args.games}")
    print(f"  MCTS sims: {args.mcts_sims}")
    print(f"  Train epochs: {args.train_epochs}")
    print(f"  Eval games: {args.eval_games}")
    print(f"{'='*70}\n")
    
    # Check if baseline exists
    if not os.path.exists(args.baseline):
        print(f"❌ Error: Baseline model not found: {args.baseline}")
        print(f"\nAvailable models:")
        for f in sorted(os.listdir('.')):
            if f.endswith('.pth'):
                size = os.path.getsize(f) / (1024 * 1024)
                print(f"  - {f} ({size:.1f} MB)")
        return
    
    # Create trainer
    trainer = Phase2Trainer(use_modal=use_modal, baseline_model=args.baseline)
    
    # Run iterations
    start_time = time.time()
    
    try:
        for i in range(args.iterations):
            trainer.run_iteration(
                games=args.games,
                mcts_sims=args.mcts_sims,
                train_epochs=args.train_epochs,
                eval_games=args.eval_games
            )
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Print summary
    trainer.print_summary()
    
    print(f"Total Time: {total_time/3600:.1f} hours")
    print(f"\n{'='*70}")
    print(f"✅ Phase 2 Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

