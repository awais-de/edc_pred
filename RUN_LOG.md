# Training Run Log & Metadata Structure

## Overview

Every training run automatically stores comprehensive metadata in `experiments/{model}_{timestamp}/metadata.json`. This enables full reproducibility and easy comparison across runs.

## What Gets Stored (Per Run)

Each run directory contains:

### Files
- **metadata.json** - Complete run configuration and results (detailed below)
- **predictions.npy** - Model predictions on test set
- **targets.npy** - Ground truth targets for test set
- **scaler_X.pkl** - Input feature scaler
- **scaler_y.pkl** - Target (EDC) scaler
- **checkpoints/best_model.ckpt** - Best model checkpoint (saved by validation loss)
- **tensorboard_logs/** - TensorBoard logs for visualization

### metadata.json Structure

```json
{
  "model_name": "hybrid_v2",
  "timestamp": "20260122_020157",
  
  "model_parameters": {
    "total": 197885568,
    "trainable": 197885568
  },
  
  "training_config": {
    "max_epochs": 200,
    "actual_epochs": 121,
    "batch_size": 2,
    "eval_batch_size": 2,
    "learning_rate": 0.001,
    "training_duration_seconds": 1593840,
    "training_duration_minutes": 26564
  },
  
  "loss_config": {
    "loss_type": "auxiliary",
    "gradient_clip_val": null,
    "aux_weight": 0.3
  },
  
  "precision_config": {
    "precision": 16,
    "mixed_precision_enabled": true
  },
  
  "data_loader_config": {
    "num_workers": 8,
    "pin_memory": true,
    "persistent_workers": true,
    "scaler_type": "standard"
  },
  
  "data_config": {
    "num_samples": 17639,
    "input_dim": 16,
    "output_length": 96000,
    "train_size": 3600,
    "val_size": 1200,
    "test_size": 1200,
    "train_ratio": 0.6,
    "val_ratio": 0.2
  },
  
  "early_stopping_config": {
    "enabled": true,
    "patience": 50,
    "stopped_at_epoch": 121
  },
  
  "metrics": {
    "overall_edc": {
      "mae": 0.250203,
      "rmse": 0.896025,
      "r2": -97.985860
    },
    "edt": {
      "mae": 0.072854,
      "rmse": 0.090517,
      "r2": -1.840263
    },
    "t20": {
      "mae": 0.726452,
      "rmse": 0.887151,
      "r2": -2.027245
    },
    "c50": {
      "mae": null,
      "rmse": null,
      "r2": null
    }
  },
  
  "best_model_path": "experiments/hybrid_v2_20260122_020157/checkpoints/best_model.ckpt"
}
```

## Comparing Runs

### Option 1: Using the Compare Script

```bash
# Print comparison table sorted by duration
python scripts/compare_runs.py

# Sort by best T20 MAE
python scripts/compare_runs.py --sort-by t20_mae

# Export to CSV for spreadsheet analysis
python scripts/compare_runs.py --export csv

# Export to JSON for programmatic analysis
python scripts/compare_runs.py --export json
```

### Option 2: Manual Inspection

Each run's metadata.json is self-contained:

```bash
# View latest run's config
cat experiments/hybrid_v2_20260122_*/metadata.json | python -m json.tool

# Compare two specific runs
diff <(jq .metrics experiments/hybrid_v2_20260122_020157/metadata.json) \
     <(jq .metrics experiments/hybrid_v2_20260122_030000/metadata.json)
```

## Key Metadata Fields to Track

### For Performance Analysis
- `metrics.overall_edc.mae` - Main objective
- `metrics.edt.mae` - EDT target (should be ≤ 0.020)
- `metrics.t20.mae` - T20 target (should be ≤ 0.020)
- `metrics.c50.mae` - C50 target (should be ≤ 0.90)

### For Reproducibility
- `training_config.batch_size` - Batch size
- `training_config.actual_epochs` - How many epochs actually ran
- `loss_config.loss_type` - Loss function used
- `loss_config.gradient_clip_val` - Gradient clipping (if any)
- `precision_config.precision` - 16-bit or 32-bit
- `data_loader_config.scaler_type` - Scaler (minmax, standard, robust)

### For Efficiency Analysis
- `training_config.training_duration_minutes` - Wall-clock training time
- `training_config.batch_size` - Batch size (larger = faster but more memory)
- `data_loader_config.num_workers` - DataLoader workers

## Example: Analyzing the 7-Hour Run

The 442-minute (7.37 hour) auxiliary loss run stored:

```json
{
  "training_duration_minutes": 442.24,
  "batch_size": 2,
  "loss_type": "auxiliary",
  "actual_epochs": 121,
  "gradient_clip_val": null,
  "precision": 16,
  "metrics": {
    "overall_edc": {"mae": 0.250203},
    "t20": {"mae": 0.726452},
    "c50": {"mae": null}
  }
}
```

This reveals:
- **Small batch size (2)** + **4800 batches/epoch** = 7+ hours
- **No gradient clipping** → numerical instability with auxiliary loss
- **Low precision (16-bit)** + complex loss numerics → gradient explosion
- **Result:** Model degraded (MAE 0.250 vs expected ~0.002)

## Best Practices

1. **Always check metadata.json after a run** to verify:
   - Actual epochs (did early stopping trigger?)
   - Duration (is it reasonable?)
   - Loss type and parameters (are they what you intended?)

2. **Use compare_runs.py regularly** to:
   - Track improvement over time
   - Identify which configurations work best
   - Spot performance regressions

3. **Keep a notes file** linking run IDs to experiments:
   ```
   20260122_020157 - Auxiliary loss test (too long, model degraded)
   20260122_030000 - Weighted EDC, batch 8, gradient clip 1.0 (GOOD)
   ```

4. **Archive good runs** before cleaning up experiments:
   ```bash
   tar -czf archive/run_20260122_030000.tar.gz experiments/hybrid_v2_20260122_030000/
   ```

## Next Steps

After each run:
1. Check `metadata.json` for basic stats
2. Run `python scripts/compare_runs.py` to see how this run ranks
3. If good results, document the run ID and command in your notes
4. If bad results, identify the problematic parameters from metadata
