#!/bin/bash
# Monitor aggressive training progress

echo "=================================================================="
echo "🔍 TRAINING STATUS CHECK"
echo "=================================================================="
echo ""

# Check if training script is running
if pgrep -f "aggressive_training.py" > /dev/null; then
    echo "✅ Aggressive training script IS RUNNING"
else
    echo "⚠️  Aggressive training script NOT running"
fi

echo ""
echo "📊 Generated files:"
ls -lh selfplay_v*.npz chess_model_sp_v*.pth 2>/dev/null | tail -10 || echo "  No new files yet"

echo ""
echo "📝 Last 15 lines of training log:"
echo "------------------------------------------------------------------"
tail -15 aggressive_training_log.txt 2>/dev/null || echo "  Log file not created yet"

echo ""
echo "=================================================================="
echo "Run this script again in a few minutes to see progress"
echo "=================================================================="

