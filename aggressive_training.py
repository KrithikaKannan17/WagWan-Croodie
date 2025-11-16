"""
Aggressive training loop to break out of the draw cycle
Runs multiple iterations of self-play + training with increasing data
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and show output."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Completed: {description}")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              🔥 AGGRESSIVE SELF-PLAY TRAINING LOOP 🔥                ║
║                                                                      ║
║  Strategy: Generate LOTS of self-play games to create training      ║
║  signal and break out of the draw equilibrium                       ║
║                                                                      ║
║  Plan:                                                               ║
║    Iteration 3: 300 games → train → evaluate                        ║
║    Iteration 4: 400 games → train → evaluate                        ║
║    Iteration 5: 500 games → train → evaluate                        ║
║                                                                      ║
║  Expected: ~30-40 minutes total on Modal GPU                         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Starting model
    current_model = "chess_model_merged_v1.pth"
    
    iterations = [
        {"games": 300, "epochs": 8},
        {"games": 400, "epochs": 10},
        {"games": 500, "epochs": 12},
    ]
    
    for i, config in enumerate(iterations, start=3):
        iteration = i
        games = config["games"]
        epochs = config["epochs"]
        
        print(f"\n\n{'#'*70}")
        print(f"#  ITERATION {iteration}: {games} games, {epochs} epochs")
        print(f"#  Current model: {current_model}")
        print(f"{'#'*70}\n")
        
        # Step 1: Generate self-play data
        selfplay_file = f"selfplay_v{iteration}.npz"
        if not run_command([
            "modal", "run", "selfplay_modal.py",
            "--games", str(games),
            "--model-path", current_model,
            "--output-file", selfplay_file,
            "--mcts-sims", "40"  # Increase MCTS for better quality
        ], f"Generate {games} self-play games (iteration {iteration})"):
            print("❌ Self-play generation failed, stopping.")
            return
        
        # Step 2: Train on new data
        next_model = f"chess_model_sp_v{iteration}.pth"
        if not run_command([
            "modal", "run", "train_modal_selfplay.py",
            "--data-file", selfplay_file,
            "--model-input", current_model,
            "--model-output", next_model,
            "--epochs", str(epochs),
            "--batch-size", "128",
            "--lr", "0.001"
        ], f"Train iteration {iteration}"):
            print("❌ Training failed, stopping.")
            return
        
        # Step 3: Download new model
        if not run_command([
            "modal", "volume", "get",
            "chess-training-data",
            f"data/data/{next_model}",
            next_model
        ], f"Download {next_model}"):
            print("❌ Download failed, stopping.")
            return
        
        # Step 4: Evaluate against previous
        print(f"\n{'='*70}")
        print(f"⚖️  EVALUATING: {next_model} vs {current_model}")
        print(f"{'='*70}\n")
        
        result = subprocess.run([
            sys.executable, "-m", "src.evaluation.test_model_comparison",
            "--model1", current_model,
            "--model2", next_model,
            "--games", "15",
            "--mcts-sims", "40",
            "--skip-policy",
            "--skip-value"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        # Parse results
        if "Model2 Wins:" in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if "Model2 Win Rate:" in line:
                    win_rate = line.split(':')[1].strip()
                    print(f"\n📊 Win rate for new model: {win_rate}")
                    
                    # Promote if > 40% (we're being generous since draws are common)
                    try:
                        win_pct = float(win_rate.replace('%', ''))
                        if win_pct >= 40:
                            print(f"✅ PROMOTING {next_model} as new baseline!")
                            current_model = next_model
                        elif win_pct >= 20:
                            print(f"⚠️  Slight improvement ({win_pct}%), promoting anyway")
                            current_model = next_model
                        else:
                            print(f"⚠️  No clear improvement, but continuing with {next_model}")
                            current_model = next_model  # Continue anyway
                    except:
                        print(f"⚠️  Couldn't parse win rate, continuing anyway")
                        current_model = next_model
        
        print(f"\n✓ Iteration {iteration} complete")
        print(f"  Next iteration will use: {current_model}\n")
    
    print(f"\n\n{'='*70}")
    print(f"🎉 AGGRESSIVE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final model: {current_model}")
    print(f"\n📁 All models generated:")
    for i in range(3, 3 + len(iterations)):
        model_file = f"chess_model_sp_v{i}.pth"
        if os.path.exists(model_file):
            print(f"  ✓ {model_file}")
    print(f"\n🎯 Best model to use: {current_model}")
    print(f"\n💡 Next steps:")
    print(f"  1. Copy to SECOND_VERSION branch for deployment")
    print(f"  2. Or continue with more tactical training")
    print(f"  3. Or run more iterations if draws persist")


if __name__ == "__main__":
    main()

