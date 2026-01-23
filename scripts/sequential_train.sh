#!/bin/bash
# Sequential training script for testing different weighted_edc configurations
# Run 3 experiments one after another (use if parallel training fails due to memory)

set -e  # Exit on error

# Base command shared across all runs
BASE_CMD="python train_model.py --model hybrid_v2 --max-samples 17639 --max-epochs 200 \
  --loss-type weighted_edc --batch-size 8 --eval-batch-size 8 \
  --num-workers 8 --pin-memory --persistent-workers \
  --precision 16 --gradient-clip-val 1.0 \
  --scaler-type standard --early-stop-patience 50"

echo "=========================================="
echo "SEQUENTIAL TRAINING - 3 WEIGHT CONFIGURATIONS"
echo "=========================================="
echo ""
echo "Expected total duration: ~570 minutes (3 × 190 min)"
echo "Logs will be saved to: run1_conservative.log, run2_moderate.log, run3_aggressive.log"
echo ""

# Run 1: Conservative (2.5x, 2.0x)
echo "▶ Starting Run 1/3: Conservative (EDT=1.5, T20=2.5, C50=2.0)"
echo "=========================================="
# Log to file and stdout simultaneously
$BASE_CMD --edt-weight 1.5 --t20-weight 2.5 --c50-weight 2.0 2>&1 | tee run1_conservative.log
echo ""
echo "✅ Run 1 complete!"
echo ""

# Run 2: Moderate (3.0x, 2.5x) - RECOMMENDED
echo "▶ Starting Run 2/3: Moderate (EDT=1.5, T20=3.0, C50=2.5) - RECOMMENDED"
echo "=========================================="
# Log to file and stdout simultaneously
$BASE_CMD --edt-weight 1.5 --t20-weight 3.0 --c50-weight 2.5 2>&1 | tee run2_moderate.log
echo ""
echo "✅ Run 2 complete!"
echo ""

# Run 3: Aggressive (4.0x, 3.5x)
echo "▶ Starting Run 3/3: Aggressive (EDT=1.5, T20=4.0, C50=3.5)"
echo "=========================================="
# Log to file and stdout simultaneously
$BASE_CMD --edt-weight 1.5 --t20-weight 4.0 --c50-weight 3.5 2>&1 | tee run3_aggressive.log
echo ""
echo "✅ Run 3 complete!"
echo ""

echo "=========================================="
echo "ALL TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  python scripts/compare_runs.py --sort-by mae --top 10"
