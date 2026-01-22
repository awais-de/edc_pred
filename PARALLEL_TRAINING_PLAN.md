# PARALLEL TRAINING STRATEGY - WEIGHT OPTIMIZATION

**Date:** January 22, 2026  
**Goal:** Meet T20 MAE ≤0.020 and C50 MAE ≤0.90 targets

---

## Current Status

**Best Run:** `hybrid_v2_20260122_112451`
- Overall EDC MAE: 0.001160 ✅
- Overall R²: 0.995 ✅
- EDT MAE: 0.004 (target ≤0.020) ✅
- T20 MAE: 0.065 (target ≤0.020) ❌ **3.2× over**
- C50 MAE: 1.451 (target ≤0.90) ❌ **1.6× over**

**Configuration:** weighted_edc with 1.5× weights insufficient

---

## Strategy: Test 3 Weight Configurations in Parallel

### Setup Complete ✅

1. **Added CLI flags to train_model.py:**
   - `--edt-weight` (default 1.5)
   - `--t20-weight` (default 1.5)
   - `--c50-weight` (default 1.5)

2. **Created parallel training script:**
   - `scripts/parallel_train.sh` - Runs 3 experiments simultaneously
   - `scripts/sequential_train.sh` - Fallback if GPU memory insufficient
   - `scripts/check_gpu.py` - Checks GPU resources

3. **Updated .gitignore:**
   - Added parallel training log files

---

## Three Experiments

### Run 1: Conservative (Baseline Improvement)
```bash
--edt-weight 1.5 --t20-weight 2.5 --c50-weight 2.0
```
- **Rationale:** Modest increase (1.5× → 2.5×/2.0×)
- **Expected:** T20 MAE ~0.040-0.050, C50 MAE ~1.0-1.2
- **Risk:** Low (proven stability)

### Run 2: Moderate (RECOMMENDED)
```bash
--edt-weight 1.5 --t20-weight 3.0 --c50-weight 2.5
```
- **Rationale:** 2× increase on T20, 1.67× on C50
- **Expected:** T20 MAE ~0.020-0.025, C50 MAE ~0.85-0.95
- **Risk:** Low-Medium (balanced approach)

### Run 3: Aggressive (Maximum Push)
```bash
--edt-weight 1.5 --t20-weight 4.0 --c50-weight 3.5
```
- **Rationale:** Strong push (2.67× on T20, 2.33× on C50)
- **Expected:** T20 MAE ~0.015-0.020, C50 MAE ~0.70-0.85
- **Risk:** Medium (may degrade overall R² slightly)

---

## Execution Instructions

### Option 1: Check GPU Resources First
```bash
# On your cluster/server:
ssh muaw1874@makalu56.mines.edu
cd /home/muaw1874/Desktop/adsp_proj/edc_pred
source venv/bin/activate

# Check GPU availability
python scripts/check_gpu.py
```

**Interpretation:**
- Exit code 0: ✅ Parallel training feasible
- Exit code 1: ⚠️ Tight but possible
- Exit code 2: ❌ Use sequential instead

### Option 2A: Launch Parallel Training (if GPU sufficient)
```bash
./scripts/parallel_train.sh
```

**Characteristics:**
- ✅ Fastest: All 3 runs in ~190 minutes
- ⚠️ Requires: ~24GB free GPU memory
- 📊 Logs: `run1_conservative.log`, `run2_moderate.log`, `run3_aggressive.log`

**Monitor progress:**
```bash
# In separate terminals:
tail -f run1_conservative.log
tail -f run2_moderate.log
tail -f run3_aggressive.log
```

### Option 2B: Launch Sequential Training (if GPU limited)
```bash
./scripts/sequential_train.sh
```

**Characteristics:**
- ✅ Safe: Guaranteed to work
- ⏱️ Slower: 3 × 190 = ~570 minutes (~9.5 hours)
- 📊 Output: Direct to terminal

---

## After Training Completes

### Compare All Runs
```bash
python scripts/compare_runs.py --sort-by mae --top 10
```

### Expected Outcomes

**Best Case:** Run 2 or Run 3 meets targets
- T20 MAE ≤0.020 ✅
- C50 MAE ≤0.90 ✅
- Overall R² ≥0.98 ✅
- **Result:** TARGETS MET! 🎉

**Good Case:** One metric meets target, other close
- T20 MAE ~0.020-0.025 (close)
- C50 MAE ~0.85-0.95 (close or met)
- **Next:** Fine-tune best performing weight config

**Needs Adjustment:** Neither metric meets target
- T20 MAE >0.025
- C50 MAE >1.0
- **Next:** Try auxiliary loss (ultra-safe settings) or further increase weights

---

## Technical Details

### Resource Requirements (Per Run)
- **GPU Memory:** ~7-8 GB
- **Training Time:** ~190 minutes (200 epochs)
- **Disk Space:** ~500 MB per run (checkpoints + results)

### Parallel Training Requirements
- **Total GPU Memory:** ~24 GB free
- **Typical Setup:** Single RTX 3090 (24GB) or A100 (40GB)
- **Alternative:** Multiple GPUs (CUDA_VISIBLE_DEVICES)

### Configuration Saved in Metadata
All runs will save weight configuration in `metadata.json`:
```json
"loss_config": {
  "loss_type": "weighted_edc",
  "edt_weight": 1.5,
  "t20_weight": 3.0,
  "c50_weight": 2.5,
  "gradient_clip_val": 1.0
}
```

---

## Troubleshooting

### Parallel Training Fails (OOM)
```bash
# Kill all runs
pkill -f train_model.py

# Use sequential instead
./scripts/sequential_train.sh
```

### One Run Fails, Others Continue
- Check specific log file: `cat run*_*.log`
- Failed run will show error, successful ones continue
- Compare successful runs only

### All Runs Succeed but Results Similar
- If Run 1 ≈ Run 2 ≈ Run 3: weights may be saturated
- Next step: Try auxiliary loss approach
- Or analyze where T20/C50 errors concentrate (specific frequency ranges)

---

## Next Actions

1. ✅ Code modifications complete
2. ✅ Training scripts created
3. ⏳ **YOU ARE HERE:** Execute training
   ```bash
   # On makalu56:
   ./scripts/parallel_train.sh
   # OR
   ./scripts/sequential_train.sh
   ```
4. ⏳ Wait ~190 min (parallel) or ~570 min (sequential)
5. ⏳ Compare results: `python scripts/compare_runs.py`
6. ⏳ Analyze best configuration and iterate if needed
