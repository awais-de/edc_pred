# Training Scripts Directory

Collection of scripts for training, monitoring, and analyzing EDC prediction models.

---

## Training Scripts

### `parallel_train.sh` ⚡
Run 3 training experiments in parallel with different weight configurations.

**Usage:**
```bash
./scripts/parallel_train.sh
```

**Requirements:**
- ~24GB free GPU memory
- Single GPU or multi-GPU setup
- Takes ~190 minutes for all 3 runs

**What it does:**
- Run 1: Conservative weights (EDT=1.5, T20=2.5, C50=2.0)
- Run 2: Moderate weights (EDT=1.5, T20=3.0, C50=2.5) - RECOMMENDED
- Run 3: Aggressive weights (EDT=1.5, T20=4.0, C50=3.5)

---

### `sequential_train.sh` 🐢
Run 3 training experiments one after another (safer, slower).

**Usage:**
```bash
./scripts/sequential_train.sh
```

**Requirements:**
- ~8GB free GPU memory
- Takes ~570 minutes (9.5 hours) total

**When to use:**
- GPU memory < 24GB
- Parallel training causes OOM
- Want guaranteed completion

---

## Monitoring Scripts

### `check_gpu.py` 🔍
Check if your GPU has sufficient memory for parallel training.

**Usage:**
```bash
python scripts/check_gpu.py
```

**Exit codes:**
- `0`: ✅ Parallel training feasible (≥24GB free)
- `1`: ⚠️ Tight but possible (≥20GB free)
- `2`: ❌ Use sequential (<20GB free)
- `3`: ❌ No GPU detected

---

### `monitor_training.py` 📊
Real-time monitoring of parallel training progress.

**Usage:**
```bash
python scripts/monitor_training.py
```

**Features:**
- Shows current epoch for each run
- Displays latest val_loss and val_mae
- Auto-refreshes every 30 seconds
- Detects completion or failures

**Alternative:** Manually tail logs
```bash
tail -f run1_conservative.log
tail -f run2_moderate.log
tail -f run3_aggressive.log
```

---

## Analysis Scripts

### `compare_runs.py` 📈
Compare all training runs with full metrics extraction.

**Usage:**
```bash
python scripts/compare_runs.py --sort-by mae --top 10
```

**Options:**
- `--sort-by`: Sort by metric (mae, rmse, r2)
- `--top N`: Show top N runs
- `--export csv`: Export to experiments/runs_comparison_*.csv

**Features:**
- Computes metrics from predictions.npy/targets.npy
- Includes EDT/T20/C50 breakdowns
- Shows training configuration from metadata.json
- Exports to CSV/JSON

---

### `extract_metrics.py` 📊
Extract metrics for all complete runs (simpler than compare_runs).

**Usage:**
```bash
python scripts/extract_metrics.py
```

**Output:**
- Console table with all runs
- CSV export to experiments/results_summary.csv

---

### `check_results.py` 🔎
Quick audit of which runs have complete results.

**Usage:**
```bash
python scripts/check_results.py
```

**Shows:**
- Which runs have metadata.json
- Which have predictions.npy/targets.npy
- Which have checkpoints
- Summary count of complete runs

---

### `list_runs.py`
List all experiment directories with timestamps.

**Usage:**
```bash
python scripts/list_runs.py
```

---

### `diagnose_experiments.py` 🔧
Detailed diagnostics for debugging training issues.

**Usage:**
```bash
python scripts/diagnose_experiments.py
```

---

## Training Utilities

### `train_edc.py`
Legacy training script (use `train_model.py` in root instead).

### `evaluate_edc.py`
Standalone evaluation script for saved models.

**Usage:**
```bash
python scripts/evaluate_edc.py --model-dir experiments/hybrid_v2_20260122_112451
```

---

## Workflow Example

### 1. Check Resources
```bash
python scripts/check_gpu.py
```

### 2. Launch Training
```bash
# If GPU sufficient:
./scripts/parallel_train.sh

# If GPU limited:
./scripts/sequential_train.sh
```

### 3. Monitor Progress
```bash
# Option A: Automated monitoring
python scripts/monitor_training.py

# Option B: Manual logs
tail -f run2_moderate.log
```

### 4. Compare Results
```bash
python scripts/compare_runs.py --sort-by mae --top 10
```

### 5. Analyze Best Run
```bash
# Check detailed metrics
python scripts/evaluate_edc.py --model-dir experiments/hybrid_v2_TIMESTAMP

# Or check files
ls -lh experiments/hybrid_v2_TIMESTAMP/
cat experiments/hybrid_v2_TIMESTAMP/metadata.json
```

---

## Files Generated

### During Training
- `run1_conservative.log` - Full output of conservative run
- `run2_moderate.log` - Full output of moderate run
- `run3_aggressive.log` - Full output of aggressive run
- `experiments/hybrid_v2_TIMESTAMP/` - Results directory for each run

### During Analysis
- `experiments/results_summary.csv` - All metrics from extract_metrics.py
- `experiments/runs_comparison_TIMESTAMP.csv` - Full comparison export
- `experiments/runs_comparison_TIMESTAMP.json` - JSON export

All generated files are .gitignored automatically.

---

## Troubleshooting

### "Command not found: python"
Use `python3` instead:
```bash
python3 scripts/check_gpu.py
python3 scripts/monitor_training.py
```

### "CUDA out of memory" during parallel training
1. Kill all runs: `pkill -f train_model.py`
2. Use sequential: `./scripts/sequential_train.sh`

### Parallel runs don't start
Check that venv is activated:
```bash
source venv/bin/activate
which python  # Should show venv path
```

### Can't find results
Runs save to `experiments/hybrid_v2_TIMESTAMP/`:
```bash
ls -lt experiments/ | head -20
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Check GPU | `python scripts/check_gpu.py` |
| Train (parallel) | `./scripts/parallel_train.sh` |
| Train (sequential) | `./scripts/sequential_train.sh` |
| Monitor progress | `python scripts/monitor_training.py` |
| Compare all runs | `python scripts/compare_runs.py` |
| Quick metrics | `python scripts/extract_metrics.py` |
| Check completeness | `python scripts/check_results.py` |
