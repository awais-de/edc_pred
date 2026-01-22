#!/bin/bash
# Parallel training script for testing different weighted_edc configurations
# Run 3 experiments with different T20/C50 weight settings

set -e  # Exit on error

# Base command shared across all runs
BASE_CMD="python train_model.py --model hybrid_v2 --max-samples 17639 --max-epochs 200 \
  --loss-type weighted_edc --batch-size 8 --eval-batch-size 8 \
  --num-workers 8 --pin-memory --persistent-workers \
  --precision 16 --gradient-clip-val 1.0 \
  --scaler-type standard --early-stop-patience 50"

echo "=========================================="
echo "PARALLEL TRAINING - 3 WEIGHT CONFIGURATIONS"
echo "=========================================="
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Cannot check GPU availability."
    echo "Proceeding anyway, but parallel training may fail if resources are insufficient."
else
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    # Get free memory (assuming single GPU)
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    
    # Each training run needs ~7-8GB. For parallel, need ~24GB free
    MIN_REQUIRED=20000  # 20GB minimum
    
    if [ "$FREE_MEM" -lt "$MIN_REQUIRED" ]; then
        echo "⚠️  Warning: Only ${FREE_MEM}MB free GPU memory."
        echo "   Parallel training may fail. Consider running sequentially."
        echo "   Press Ctrl+C to cancel, or wait 5 seconds to continue..."
        sleep 5
    else
        echo "✅ Sufficient GPU memory available (${FREE_MEM}MB free)"
    fi
fi

echo ""
echo "Starting 3 parallel training runs:"
echo "  Run 1 (Conservative): EDT=1.5, T20=2.5, C50=2.0"
echo "  Run 2 (Moderate):     EDT=1.5, T20=3.0, C50=2.5"
echo "  Run 3 (Aggressive):   EDT=1.5, T20=4.0, C50=3.5"
echo ""
echo "Expected duration: ~190 minutes per run (parallel)"
echo "Logs will be saved to: experiments/hybrid_v2_*/tensorboard_logs/"
echo ""

# Run 1: Conservative (2.5x, 2.0x)
echo "▶ Launching Run 1 (Conservative)..."
$BASE_CMD --edt-weight 1.5 --t20-weight 2.5 --c50-weight 2.0 > run1_conservative.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

# Run 2: Moderate (3.0x, 2.5x) - RECOMMENDED
echo "▶ Launching Run 2 (Moderate - RECOMMENDED)..."
$BASE_CMD --edt-weight 1.5 --t20-weight 3.0 --c50-weight 2.5 > run2_moderate.log 2>&1 &
PID2=$!
echo "  PID: $PID2"

# Run 3: Aggressive (4.0x, 3.5x)
echo "▶ Launching Run 3 (Aggressive)..."
$BASE_CMD --edt-weight 1.5 --t20-weight 4.0 --c50-weight 3.5 > run3_aggressive.log 2>&1 &
PID3=$!
echo "  PID: $PID3"

echo ""
echo "All runs launched! Monitoring progress..."
echo ""
echo "To monitor individual runs:"
echo "  tail -f run1_conservative.log"
echo "  tail -f run2_moderate.log"
echo "  tail -f run3_aggressive.log"
echo ""
echo "To stop all runs: kill $PID1 $PID2 $PID3"
echo ""

# Wait for all runs to complete
echo "Waiting for all runs to complete..."
echo "(This will take approximately 190 minutes)"
echo ""

FAILED=0

# Wait for Run 1
wait $PID1
STATUS1=$?
if [ $STATUS1 -eq 0 ]; then
    echo "✅ Run 1 (Conservative) completed successfully"
else
    echo "❌ Run 1 (Conservative) failed with exit code $STATUS1"
    FAILED=1
fi

# Wait for Run 2
wait $PID2
STATUS2=$?
if [ $STATUS2 -eq 0 ]; then
    echo "✅ Run 2 (Moderate) completed successfully"
else
    echo "❌ Run 2 (Moderate) failed with exit code $STATUS2"
    FAILED=1
fi

# Wait for Run 3
wait $PID3
STATUS3=$?
if [ $STATUS3 -eq 0 ]; then
    echo "✅ Run 3 (Aggressive) completed successfully"
else
    echo "❌ Run 3 (Aggressive) failed with exit code $STATUS3"
    FAILED=1
fi

echo ""
echo "=========================================="
echo "PARALLEL TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  Run 1 (Conservative): Exit code $STATUS1"
echo "  Run 2 (Moderate):     Exit code $STATUS2"
echo "  Run 3 (Aggressive):   Exit code $STATUS3"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All runs completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  python scripts/compare_runs.py --sort-by mae --top 10"
else
    echo "⚠️  Some runs failed. Check logs:"
    echo "  cat run1_conservative.log"
    echo "  cat run2_moderate.log"
    echo "  cat run3_aggressive.log"
fi

exit $FAILED
