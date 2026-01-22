# MASTER PROJECT CONTEXT DOCUMENT
**Complete EDC Prediction Training Project State**  
**Generated:** January 22, 2026  
**Purpose:** Full context transfer for continuation in remote VS Code session

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Problem Analysis & Root Causes](#problem-analysis--root-causes)
3. [Solutions Implemented](#solutions-implemented)
4. [Code Changes Summary](#code-changes-summary)
5. [Current State & Results](#current-state--results)
6. [Training Strategy](#training-strategy)
7. [Scripts Overview](#scripts-overview)
8. [Next Steps](#next-steps)
9. [Troubleshooting](#troubleshooting)

---

## PROJECT OVERVIEW

### Objectives
- **Primary:** Predict EDC (Energy Decay Curve) from room features
- **Architecture:** Hybrid LSTM + CNN model (~198M parameters)
- **Dataset:** 6000 aligned EDC samples + room features, 60/20/20 train/val/test split
- **Targets:**
  - Overall EDC MAE ≤ 0.020
  - EDT (Early Decay Time) MAE ≤ 0.020 ✅
  - T20 MAE ≤ 0.020 ❌
  - C50 (Clarity) MAE ≤ 0.90 ❌
  - Overall R² ≥ 0.98 ✅

### Current Status: 2/4 Targets Met
- ✅ EDT MAE: 0.004 (5× better than target)
- ✅ Overall R²: 0.995 (target: 0.98)
- ❌ T20 MAE: 0.065 (3.2× over target)
- ❌ C50 MAE: 1.451 (1.6× over target)

---

## PROBLEM ANALYSIS & ROOT CAUSES

### Issue 1: 7-Hour Training Failure (Batch Size 2)

**Symptoms:**
- Training took 442 minutes (7+ hours)
- Val_loss: 815 (epoch 70) → 1380+ (epoch 120) - diverged catastrophically
- Overall MAE degraded from ~0.002 to 0.250 (125× worse)
- Run: `hybrid_v2_20260122_020157`

**Root Causes:**
1. **Batch size 2** caused 4× more batches per epoch (2400 vs 600)
   - Normal: 600 batches/epoch × 111 epochs = 67k total batches
   - Batch 2: 2400 batches/epoch × 51 epochs = 122k total batches
   - 6× longer total training time

2. **Auxiliary loss instability**
   - Original softmax temperature: 2.0 (too aggressive)
   - Mixed precision (16-bit) + complex loss = numerical instability
   - Loss component imbalance (aux_weight default 0.3 too high)
   - **No gradient clipping** = gradient explosion risk

3. **Configuration Issues**
   - Early stop patience 50 allowed 51 extra epochs before triggering
   - Training continued despite degradation

### Issue 2: T20/C50 Metrics Over Target

**Symptoms:**
- Best run (`hybrid_v2_20260122_112451`):
  - Overall MAE: 0.001160 (excellent)
  - T20 MAE: 0.065 (need 0.020) - 3.2× over
  - C50 MAE: 1.451 (need 0.90) - 1.6× over

**Root Cause:**
- Weighted EDC loss with 1.5× weights prioritizes overall EDC fit
- Doesn't provide strong enough signal for acoustic parameter optimization
- T20/C50 are harder to predict with generic weighting

---

## SOLUTIONS IMPLEMENTED

### Solution 1: Stabilized AuxiliaryAcousticLoss

**File:** `src/models/lstm_model.py` (AuxiliaryAcousticLoss class)

**Changes:**
- Reduced softmax temperature: 2.0 → 1.0 (gentler gradient flow)
- Stricter clamping:
  - edc_db: clamped to [-60, 0]
  - t20_raw: max reduced from 10 to 5
  - c50: clamped to [-50, 50]
- Reduced default aux_weight: 0.3 → 0.1 (lower loss component imbalance)
- Added input clamping and loss component limits (max 1e4)
- Added Inf checking and fallback to MSE
- Normalized auxiliary loss: 0.5×(T20+C50) instead of 1.0×each

### Solution 2: Added CLI Stability Flags

**File:** `train_model.py`

**New Arguments:**
```python
--gradient-clip-val FLOAT     # Gradient clipping threshold (1.0 recommended)
--aux-weight FLOAT            # Auxiliary loss weight (0.1 recommended)
--no-mixed-precision          # Force 32-bit precision (disable 16-bit)
--edt-weight FLOAT            # EDT weight in weighted_edc loss (1.5 default)
--t20-weight FLOAT            # T20 weight in weighted_edc loss (1.5 default)
--c50-weight FLOAT            # C50 weight in weighted_edc loss (1.5 default)
```

**Implementation:**
- Arguments parsed and applied to model.criterion
- Saved in metadata.json for tracking
- Allows flexible weight tuning without code changes

### Solution 3: Enhanced Metadata Tracking

**File:** `train_model.py` (metadata.json saving)

**New Sections:**
```json
{
  "model_name": "hybrid_v2",
  "timestamp": "20260122_112451",
  "training_config": {
    "max_epochs": 200,
    "actual_epochs": 200,
    "batch_size": 8,
    "training_duration_minutes": 189.92
  },
  "loss_config": {
    "loss_type": "weighted_edc",
    "gradient_clip_val": 1.0,
    "edt_weight": 1.5,
    "t20_weight": 1.5,
    "c50_weight": 1.5
  },
  "precision_config": {
    "precision": 16,
    "mixed_precision_enabled": true
  }
}
```

### Solution 4: Created Run Comparison Tools

**Files Created:**
1. `scripts/check_results.py` - Audits which runs have complete files
2. `scripts/extract_metrics.py` - Computes metrics from predictions/targets
3. `scripts/compare_runs.py` (rewritten) - Unified comparison tool
4. `RUN_LOG.md` - Documentation of metadata structure

**Key Insight:** Compute metrics from predictions.npy/targets.npy (always available) rather than relying on metadata format consistency.

**Result:** Successfully extracted metrics from 18 complete runs, identified best performers and failure cases.

---

## CODE CHANGES SUMMARY

### Files Modified

#### 1. `train_model.py` - Main Training Script
**Lines Modified:**
- Lines 127-131: Added --edt-weight, --t20-weight, --c50-weight arguments
- Lines 258-263: Patched weighted_edc weights from CLI
- Lines 409-413: Saved weights to metadata.json

**Key Addition:**
```python
if args.loss_type == "weighted_edc" and hasattr(model.criterion, 'edt_weight'):
    model.criterion.edt_weight = args.edt_weight
    model.criterion.t20_weight = args.t20_weight
    model.criterion.c50_weight = args.c50_weight
```

#### 2. `src/models/lstm_model.py` - AuxiliaryAcousticLoss
**Class:** AuxiliaryAcousticLoss (lines 160-240+)

**Improvements:**
- Constructor: aux_weight default 0.3 → 0.1, clamped [0.01, 1.0]
- _compute_t20(): 
  - softmax temp 2.0 → 1.0
  - edc_db clamped [-60, 0]
  - t20_raw max reduced 10 → 5
- forward():
  - Input clamping before processing
  - MSE loss clamping (max 1e4)
  - Loss component clamping
  - Inf checking and MSE fallback
  - Normalized auxiliary as 0.5×(T20+C50)

#### 3. `.gitignore` - Updated Ignore Patterns
**Added:**
```
run1_conservative.log
run2_moderate.log
run3_aggressive.log
```

### Files Created

#### 1. `scripts/parallel_train.sh` (Executable)
- Launches 3 training runs simultaneously
- Requires ~24GB free GPU memory
- Each run: ~190 minutes
- Total: ~190 minutes (parallel)

#### 2. `scripts/sequential_train.sh` (Executable)
- Launches 3 training runs one after another
- Requires ~8GB free GPU memory (safe)
- Each run: ~190 minutes
- Total: ~570 minutes (~9.5 hours)

#### 3. `scripts/check_gpu.py`
- Checks GPU resources
- Determines if parallel training feasible
- Exit codes: 0 (feasible), 1 (tight), 2 (use sequential), 3 (no GPU)

#### 4. `scripts/monitor_training.py`
- Real-time monitoring of parallel/sequential training
- Shows epoch, val_loss, val_mae for each run
- Auto-refreshes every 30 seconds
- Detects completion or failures

#### 5. `PARALLEL_TRAINING_PLAN.md`
- Complete strategy documentation
- Configuration details for 3 experiments
- Execution instructions and troubleshooting

#### 6. `scripts/README.md`
- Reference guide for all scripts
- Usage examples and workflow

#### 7. `CONVERSATION_CONTEXT.md` (This File)
- Master context document for chat continuation

---

## CURRENT STATE & RESULTS

### Best Run So Far
**Directory:** `experiments/hybrid_v2_20260122_112451`  
**Configuration:**
- Model: hybrid_v2
- Loss: weighted_edc (EDT=1.5, T20=1.5, C50=1.5)
- Batch size: 8
- Precision: 16-bit
- Gradient clipping: 1.0
- Epochs: 200 (full, no early stop)
- Training time: 189.92 minutes (~3.2 hours)

**Metrics:**
```
Overall EDC:  MAE=0.001160  RMSE=0.006630  R²=0.9946
EDT Metrics:  MAE=0.004189  RMSE=0.006929  R²=0.9834 ✅
T20 Metrics:  MAE=0.064688  RMSE=0.113695  R²=0.9503 ❌
C50 Metrics:  MAE=1.450848  RMSE=3.254320  R²=0.7252 ❌
```

**Assessment:**
- ✅ Excellent overall EDC and EDT performance
- ✅ R² well above target (0.995 > 0.98)
- ❌ T20 MAE 3.2× over target (0.065 vs 0.020)
- ❌ C50 MAE 1.6× over target (1.451 vs 0.90)

### Previous Best Runs
- `hybrid_v2_20260121_132247`: MAE=0.001470, R²=0.668 (early stopped at 111 epochs)
- Multiple baseline runs (MSE loss): MAE=0.001500-0.002000

### Failed Run
- `hybrid_v2_20260122_020157`: Catastrophic failure (MAE=0.250)
  - Configuration: batch 2, auxiliary loss, no gradient clipping, 442 minutes
  - Root cause: See Problem Analysis section above

### Incomplete Runs
- 17 directories with only checkpoints (no predictions/targets)
- 2 LSTM runs with malformed JSON (pre-enhancement format)

---

## TRAINING STRATEGY

### Current Challenge
Weighted EDC loss with 1.5× weights achieves excellent overall fit but insufficient T20/C50 optimization.

**Analysis:**
- Need T20 MAE to improve ~3.2× (0.065 → 0.020)
- Need C50 MAE to improve ~1.6× (1.451 → 0.90)
- Overall R² must stay ≥0.98 (currently 0.995)

### Solution: Test 3 Weight Configurations

**Why this approach:**
1. ✅ Proven stable architecture (batch 8, gradient clipping 1.0)
2. ✅ Fast validation (~190 min per run)
3. ✅ Direct T20/C50 optimization via weight increase
4. ✅ Low risk (same configuration, only weights change)
5. ✅ Sequential training safe (fits in 16GB GPU)

### The 3 Experiments

| Configuration | EDT Weight | T20 Weight | C50 Weight | Expected T20 MAE | Expected C50 MAE | Risk |
|---------------|------------|------------|------------|------------------|------------------|------|
| Conservative | 1.5 | 2.5 | 2.0 | 0.040-0.050 | 1.0-1.2 | Low |
| **Moderate** | 1.5 | **3.0** | **2.5** | **0.020-0.025** | **0.85-0.95** | **Low** |
| Aggressive | 1.5 | 4.0 | 3.5 | 0.015-0.020 | 0.70-0.85 | Medium |

**Recommendation:** Run 2 (Moderate) most likely to meet targets with minimal R² tradeoff.

---

## SCRIPTS OVERVIEW

### Training Scripts

#### `scripts/parallel_train.sh` ⚡
**When to use:** GPU has ≥24GB free memory

**Command:**
```bash
./scripts/parallel_train.sh
```

**What it does:**
- Spawns 3 training processes simultaneously
- Logs to: run1_conservative.log, run2_moderate.log, run3_aggressive.log
- Waits for all to complete before exiting
- Expected time: ~190 minutes

**GPU Requirements:** Single RTX 3090 (24GB) or A100 (40GB+)

#### `scripts/sequential_train.sh` 🐢
**When to use:** GPU has <24GB free (like Quadro P5000 with 16GB)

**Command:**
```bash
./scripts/sequential_train.sh
```

**What it does:**
- Runs 3 experiments one after another
- Direct output to terminal
- Each completes before next starts
- Expected time: ~570 minutes (~9.5 hours)

**GPU Requirements:** ≥8GB (safe)

### Monitoring & Analysis Scripts

#### `scripts/check_gpu.py` 🔍
**Purpose:** Check if GPU has sufficient memory for parallel training

**Command:**
```bash
python scripts/check_gpu.py
```

**Exit codes:**
- 0: Parallel feasible (≥24GB free)
- 1: Tight but possible (≥20GB free)
- 2: Use sequential (<20GB free)
- 3: No GPU detected

#### `scripts/monitor_training.py` 📊
**Purpose:** Real-time monitoring of training progress

**Command:**
```bash
python scripts/monitor_training.py
```

**Features:**
- Shows current epoch for each run
- Displays latest val_loss and val_mae
- Auto-refreshes every 30 seconds
- Detects completion/failures
- Press Ctrl+C to stop (training continues)

#### `scripts/compare_runs.py` 📈
**Purpose:** Compare all training runs with detailed metrics

**Commands:**
```bash
# Show all runs sorted by MAE
python scripts/compare_runs.py --sort-by mae

# Show top 10 runs
python scripts/compare_runs.py --sort-by mae --top 10

# Export to CSV
python scripts/compare_runs.py --sort-by mae --export csv
```

**Output:**
- Console table with all metrics
- Exports to: experiments/runs_comparison_TIMESTAMP.csv
- Includes: EDT/T20/C50 breakdown, R², configuration details

#### `scripts/extract_metrics.py` 📊
**Purpose:** Extract metrics from all complete runs (simpler than compare_runs)

**Command:**
```bash
python scripts/extract_metrics.py
```

**Output:**
- Console table
- Exports to: experiments/results_summary.csv
- Only uses predictions.npy/targets.npy (always available)

#### `scripts/check_results.py` 🔎
**Purpose:** Quick audit of which runs have complete files

**Command:**
```bash
python scripts/check_results.py
```

**Shows:**
- Which runs have metadata.json
- Which have predictions/targets
- Which have checkpoints
- Summary: "18/35 complete"

---

## NEXT STEPS

### Step 1: Launch Training (YOU ARE HERE)
On makalu56 server:
```bash
cd /home/muaw1874/Desktop/adsp_proj/edc_pred
source venv/bin/activate

# Check GPU resources
python scripts/check_gpu.py

# Launch sequential training (16GB GPU confirmed sufficient)
./scripts/sequential_train.sh
```

**Expected duration:** ~570 minutes (~9.5 hours)

**What happens:**
- Run 1 (Conservative): ~190 min
- Run 2 (Moderate): ~190 min ← WATCH THIS ONE
- Run 3 (Aggressive): ~190 min

### Step 2: Monitor Progress
In another terminal:
```bash
# Option A: Automated monitoring
python scripts/monitor_training.py

# Option B: Manual monitoring
tail -f run2_moderate.log

# Option C: Watch all three
watch -n 30 "tail -5 run*.log"
```

### Step 3: Wait for Completion
Estimated completion time: Jan 22, 2026 ~23:00-02:00 (depending on start time)

### Step 4: Compare Results
Once all runs complete:
```bash
python scripts/compare_runs.py --sort-by mae --top 10
```

### Step 5: Analysis
Expected outcomes:
- **Best case:** Run 2 or 3 meets T20≤0.020 and C50≤0.90 targets ✅
- **Good case:** One metric meets target, other close ✅
- **Needs adjustment:** Neither meets targets → Try auxiliary loss or further increase weights

---

## TROUBLESHOOTING

### Error: "Traceback... TypeError: __init__() got an unexpected keyword argument 'capture_output'"
**Cause:** Python 3.6 doesn't support capture_output (added in 3.7)
**Solution:** Already fixed in check_gpu.py using stdout=subprocess.PIPE

### Error: "CUDA out of memory" during parallel training
**Solution:**
```bash
pkill -f train_model.py
./scripts/sequential_train.sh  # Use sequential instead
```

### Error: "Command not found: python"
**Solution:** Use `python3` instead:
```bash
python3 scripts/check_gpu.py
```

### Training seems stuck
**Solution:** Check GPU utilization:
```bash
nvidia-smi -l 1  # Refresh every 1 second
```

If GPU usage is 0%, training may have crashed. Check logs:
```bash
tail -100 run2_moderate.log | tail -20
```

### One run fails but others continue
**Solution:** Check specific log:
```bash
cat run1_conservative.log | tail -50
```

Compare successful runs only when complete.

### All runs complete but results worse than expected
**Analysis steps:**
1. Check if weights were actually applied:
   ```bash
   cat experiments/hybrid_v2_*/metadata.json | grep -A5 loss_config
   ```

2. If weights applied but results worse, consider:
   - Weights may have been at saturation point
   - T20/C50 errors concentrated in specific frequency ranges
   - May need auxiliary loss approach instead

3. Next iteration:
   ```bash
   --loss-type auxiliary --aux-weight 0.05 --no-mixed-precision --gradient-clip-val 1.0
   ```

---

## KEY CONFIGURATION PARAMETERS

### Proven Stable Configuration (Current Best)
```bash
--model hybrid_v2
--max-samples 17639
--max-epochs 200
--loss-type weighted_edc
--batch-size 8
--eval-batch-size 8
--num-workers 8
--pin-memory
--persistent-workers
--precision 16
--gradient-clip-val 1.0
--scaler-type standard
--early-stop-patience 50
```

### What NOT to Use
- ❌ batch-size 2 (causes 4× slower training and instability)
- ❌ No gradient-clip-val (risk of gradient explosion)
- ❌ Mixed precision (16-bit) without gradient clipping (numerical instability)
- ❌ High aux-weight (>0.1) with auxiliary loss (imbalance)

### Variable Parameters (To Tune)
- edt-weight: 1.5 (increase to 2.0-2.5 for EDT focus)
- t20-weight: 1.5 (increase to 2.5-4.0 to improve T20)
- c50-weight: 1.5 (increase to 2.0-3.5 to improve C50)
- gradient-clip-val: 1.0 (range 0.5-10.0)
- max-epochs: 200 (can reduce to 100-150 if time-critical)

---

## PERFORMANCE BENCHMARKS

### Training Speed
| Configuration | Batches/Epoch | Time/Epoch | Total (200ep) |
|---------------|---------------|------------|---------------|
| Batch 8, 16-bit, clipping | 600 | 55-57s | 190 min |
| Batch 8, 32-bit, clipping | 600 | 75-80s | 260 min |
| Batch 2, 16-bit, no clip | 2400 | 110s | 442 min |

### Memory Requirements
| Configuration | GPU Memory |
|---------------|-----------|
| Single batch 8 training | ~7-8 GB |
| 3 Parallel (×3) | ~24 GB |
| Inference only | ~4 GB |

---

## FINAL NOTES

### Session Summary
- **Duration:** January 22, 2026
- **Conversation Focus:** Debugging 7-hour training failure, analyzing runs, creating weight optimization strategy
- **Code Changes:** 4 files modified, 7 new scripts created, 2 documentation files
- **Current Action:** Ready to launch sequential training with 3 weight configurations

### Project Health
✅ Codebase stable and tested  
✅ All scripts executable and documented  
✅ Training infrastructure robust (gradient clipping, precision control, etc.)  
✅ Run comparison tools functional  
✅ Next iteration well-planned  

### Success Criteria for This Phase
- [ ] All 3 runs complete without errors
- [ ] At least one run achieves T20 MAE ≤ 0.020
- [ ] At least one run achieves C50 MAE ≤ 0.90
- [ ] Overall R² maintained ≥ 0.98

### If This Phase Succeeds
Next will be fine-tuning or auxiliary loss approach to address remaining gaps.

---

**End of Master Context Document**  
*Use this entire document as context for the remote VS Code chat to continue seamlessly.*
