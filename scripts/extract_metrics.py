#!/usr/bin/env python3
"""
Extract metrics from all complete runs with predictions/targets.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

experiments_dir = Path("experiments")

if not experiments_dir.exists():
    print("experiments/ directory not found")
    exit(1)

print("\n" + "="*150)
print("EXTRACTING METRICS FROM ALL COMPLETE RUNS")
print("="*150 + "\n")

results = []

for run_dir in sorted(experiments_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    
    # Check if run is complete (has all required files)
    metadata_file = run_dir / "metadata.json"
    preds_file = run_dir / "predictions.npy"
    targets_file = run_dir / "targets.npy"
    
    if not (metadata_file.exists() and preds_file.exists() and targets_file.exists()):
        continue
    
    try:
        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load predictions and targets
        predictions = np.load(preds_file)
        targets = np.load(targets_file)
        
        # Compute metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # Extract config from metadata
        model_name = metadata.get("model_name", "?")
        loss_type = metadata.get("loss_config", {}).get("loss_type", "?")
        batch_size = metadata.get("training_config", {}).get("batch_size", "?")
        epochs = metadata.get("training_config", {}).get("actual_epochs", "?")
        duration_min = metadata.get("training_config", {}).get("training_duration_minutes", "?")
        precision = metadata.get("precision_config", {}).get("precision", "?")
        
        results.append({
            "run_id": run_dir.name,
            "model": model_name,
            "loss_type": loss_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "duration_min": duration_min,
            "precision": precision,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
    
    except Exception as e:
        print(f"⚠ Error processing {run_dir.name}: {e}")

if not results:
    print("No complete runs found")
    exit(1)

# Create DataFrame
df = pd.DataFrame(results)

# Sort by MAE (best performance first)
df = df.sort_values("mae")

print(f"Found {len(df)} complete runs\n")
print(df.to_string(index=False))

print("\n" + "="*150)
print("BEST 5 RUNS (by MAE):")
print("="*150)
for idx, row in df.head(5).iterrows():
    print(f"\n{idx+1}. {row['run_id']}")
    print(f"   Model: {row['model']}, Loss: {row['loss_type']}, Epochs: {int(row['epochs'])}")
    print(f"   MAE: {row['mae']:.6f}, RMSE: {row['rmse']:.6f}, R²: {row['r2']:.6f}")

print("\n" + "="*150 + "\n")

# Export to CSV
csv_file = "experiments/results_summary.csv"
df.to_csv(csv_file, index=False)
print(f"✓ Results exported to {csv_file}")
