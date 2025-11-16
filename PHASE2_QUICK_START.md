# Phase 2: Iterative Self-Play Training - Quick Start

## 🎯 Goal
Improve your chess model through iterative self-play training (AlphaZero-style).

## 📋 Prerequisites
- ✅ Phase 1 complete with `chess_model_best.pth` (your baseline)
- ✅ Virtual environment activated (`source .venv312/bin/activate`)
- ✅ All dependencies installed
- ✅ (Optional) Modal account setup for GPU training

## 🚀 Quick Start

### Option 1: Simple 3-Iteration Loop (Recommended)

```bash
# Local training (CPU/GPU) - Good for testing
python phase2_iterative_loop.py --iterations 3 --games 100 --local

# Modal GPU training - Better quality, faster
python phase2_iterative_loop.py --iterations 3 --games 300 --modal
```

This will automatically:
1. Generate self-play games with your best model
2. Train on the new data
3. Evaluate the new model vs baseline
4. Promote the new model if it wins >55% of games
5. Repeat for 3 iterations

### Option 2: Manual Step-by-Step

If you want more control:

```bash
# 1. Generate self-play data (100 games)
python selfplay_generator.py \
  --games 100 \
  --mcts-sims 50 \
  --model chess_model_best.pth \
  --output datasets/selfplay_v1.npz

# 2. Train on self-play data
python -m src.training.train_selfplay \
  --data datasets/selfplay_v1.npz \
  --model chess_model_best.pth \
  --output chess_model_sp_v1.pth \
  --epochs 5 \
  --batch-size 128

# 3. Evaluate new model vs baseline
python -m src.evaluation.test_model_comparison \
  --model1 chess_model_best.pth \
  --model2 chess_model_sp_v1.pth \
  --games 20 \
  --mcts-sims 20

# 4. If win rate > 55%, promote:
cp chess_model_sp_v1.pth chess_model_best.pth

# 5. Repeat from step 1
```

## 🎮 Usage Examples

### Quick Test (1 iteration, minimal)
```bash
python phase2_iterative_loop.py \
  --iterations 1 \
  --games 50 \
  --train-epochs 3 \
  --eval-games 10 \
  --local
```
⏱️ Time: ~10-30 minutes

### Standard Run (3 iterations, local)
```bash
python phase2_iterative_loop.py \
  --iterations 3 \
  --games 100 \
  --mcts-sims 50 \
  --train-epochs 5 \
  --eval-games 20 \
  --local
```
⏱️ Time: ~1-3 hours

### High-Quality Run (3 iterations, Modal GPU)
```bash
python phase2_iterative_loop.py \
  --iterations 3 \
  --games 300 \
  --mcts-sims 50 \
  --train-epochs 5 \
  --eval-games 20 \
  --modal
```
⏱️ Time: ~30-60 minutes (much faster on GPU)

### Continue from Existing Model
```bash
python phase2_iterative_loop.py \
  --baseline chess_model_sp_v2.pth \
  --iterations 2 \
  --games 200 \
  --local
```

## 📊 Understanding Results

After each iteration, you'll see:

```
📊 ITERATION 1 SUMMARY
======================================================================
  Baseline: chess_model_best.pth
  New Model: chess_model_sp_v1.pth
  Win Rate: 62.5%
  Time: 15.3 minutes

  ✅ NEW MODEL IS STRONGER - Promoting as new baseline!
======================================================================
```

**Win Rate Interpretation:**
- **> 55%**: ✅ New model is stronger → promoted as baseline
- **45-55%**: ➡️ Models are similar → keep current baseline
- **< 45%**: ⚠️ New model is weaker → keep current baseline

## 🔄 The AlphaZero Loop

```
Iteration 1:
  chess_model_best.pth (baseline)
    ↓ generate 100 games
  selfplay_v1.npz
    ↓ train 5 epochs
  chess_model_sp_v1.pth
    ↓ evaluate (20 games)
  Win rate: 62% → ✅ Promote!

Iteration 2:
  chess_model_sp_v1.pth (new baseline)
    ↓ generate 100 games
  selfplay_v2.npz
    ↓ train 5 epochs
  chess_model_sp_v2.pth
    ↓ evaluate (20 games)
  Win rate: 58% → ✅ Promote!

Iteration 3:
  chess_model_sp_v2.pth (new baseline)
    ↓ generate 100 games
  selfplay_v3.npz
    ↓ train 5 epochs
  chess_model_sp_v3.pth
    ↓ evaluate (20 games)
  Win rate: 48% → ➡️ Keep v2

Final: chess_model_sp_v2.pth is your best model!
```

## 📁 Output Files

After 3 iterations, you'll have:

```
datasets/
  selfplay_v1.npz        # ~5-10 MB per 100 games
  selfplay_v2.npz
  selfplay_v3.npz

chess_model_sp_v1.pth    # ~4-5 MB each
chess_model_sp_v2.pth
chess_model_sp_v3.pth
```

## ⚙️ Hyperparameter Tuning

### For Quick Testing
```bash
--games 50            # Fewer games
--mcts-sims 20        # Less MCTS depth
--train-epochs 3      # Fewer epochs
--eval-games 10       # Quick evaluation
```

### For Better Quality
```bash
--games 300           # More diverse data
--mcts-sims 75        # Deeper search
--train-epochs 10     # More training
--eval-games 30       # More reliable evaluation
```

### Recommended Balance (Default)
```bash
--games 200           # Good data diversity
--mcts-sims 50        # Good search depth
--train-epochs 5      # Prevents overfitting
--eval-games 20       # Reliable evaluation
```

## 🐛 Troubleshooting

### Self-Play is Slow
- **Problem**: Taking >5 minutes per game on CPU
- **Solution**: 
  - Reduce `--mcts-sims` to 20-30
  - Use `--modal` for GPU acceleration
  - Or reduce `--games` to 50-100

### Model Not Improving
- **Problem**: Win rate stuck at 50%
- **Solutions**:
  - Increase `--games` to 300+ for more diverse data
  - Increase `--mcts-sims` to 75-100 for better quality
  - Check if model is overfitting (look at training logs)
  - Try more iterations (5-10 instead of 3)

### Out of Memory
- **Problem**: CUDA out of memory during training
- **Solutions**:
  - Reduce `--batch-size` (try 64 or 32)
  - Use `--modal` for cloud GPU with more memory
  - Close other GPU-using programs

### Model Getting Weaker
- **Problem**: Later iterations have lower win rates
- **Possible causes**:
  - Overfitting - reduce `--train-epochs`
  - Not enough data - increase `--games`
  - Evaluation noise - increase `--eval-games` to 30-40

## 📈 Expected Progress

**Typical improvement per iteration:**
- Iteration 1: +5-15% improvement over baseline
- Iteration 2: +3-10% improvement over iteration 1
- Iteration 3: +2-5% improvement over iteration 2
- ...diminishing returns after 5-10 iterations

**When to stop:**
- Win rate improvement < 2% for 2 consecutive iterations
- Or after 5-10 iterations
- Or when satisfied with performance

## ✅ When Phase 2 is Complete

After 3-5 successful iterations, you'll have:
- ✅ A significantly stronger model (`chess_model_sp_vX.pth`)
- ✅ High-quality self-play data
- ✅ Understanding of your model's strength progression

**Next: Move to Phase 3/4**
- Phase 3: Blend tactical data with self-play
- Phase 4: Final large-scale training on cloud GPU

## 🔗 Related Files

- `selfplay_generator.py` - Self-play data generation
- `selfplay_modal.py` - Modal GPU generation
- `src/training/train_selfplay.py` - Training script
- `train_modal_selfplay.py` - Modal GPU training
- `src/evaluation/test_model_comparison.py` - Model evaluation
- `PHASE2_GUIDE.md` - Detailed Phase 2 documentation
- `PHASE2_MODAL_GUIDE.md` - Modal-specific guide

## 💡 Pro Tips

1. **Start with a quick test** (1 iteration, 50 games) to verify everything works
2. **Use Modal for serious training** - It's much faster and produces better data
3. **Monitor training logs** - Watch for overfitting (val loss increasing)
4. **Keep all model versions** - You can always go back if later versions fail
5. **Increase eval games** if win rates seem inconsistent (try 30-40 games)
6. **Save your best model** separately as a backup

## 🎯 Success Criteria for Phase 2

Before moving to Phase 3, aim for:
- ✅ At least 2-3 successful iterations with improving models
- ✅ Final model beats baseline by >10% win rate
- ✅ Self-play games reach 30-50 average move length
- ✅ Evaluation shows consistent move quality improvement

Good luck! 🚀

