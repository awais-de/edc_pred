"""
Compare all training runs and generate a summary report.

Usage:
    python scripts/compare_runs.py [--sort-by metric] [--export csv/json]

Reads metadata.json from all experiment directories and computes metrics from predictions.npy
"""

#!/usr/bin/env python3
"""
Compare all training runs by extracting metrics from predictions/targets.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def extract_all_runs(experiments_dir="experiments"):
    """Extract metrics from all complete runs."""
    runs = []
    errors = []
    
    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return runs
    
    print(f"\nScanning {experiments_dir} for complete runs...\n")
    
    for run_dir in sorted(experiments_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Check if run is complete
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
            
            # Extract config
            run_info = {
                "run_id": run_dir.name,
                "model": metadata.get("model_name"),
                "loss_type": metadata.get("loss_config", {}).get("loss_type"),
                "batch_size": metadata.get("training_config", {}).get("batch_size"),
                "epochs": metadata.get("training_config", {}).get("actual_epochs"),
                "duration_min": metadata.get("training_config", {}).get("training_duration_minutes"),
                "precision": metadata.get("precision_config", {}).get("precision"),
                "gradient_clip": metadata.get("loss_config", {}).get("gradient_clip_val"),
                "aux_weight": metadata.get("loss_config", {}).get("aux_weight"),
                "scaler_type": metadata.get("data_loader_config", {}).get("scaler_type"),
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }
            
            runs.append(run_info)
        
        except json.JSONDecodeError as e:
            errors.append(f"  ⚠ {run_dir.name}: JSON parsing error")
        except Exception as e:
            errors.append(f"  ⚠ {run_dir.name}: {type(e).__name__}")
    
    if errors:
        print("Skipped malformed runs:")
        for error in errors:
            print(error)
        print()
    
    return runs


def print_comparison(runs, sort_by="mae"):
    """Print comparison table."""
    if not runs:
        print("No complete runs found")
        return
    
    df = pd.DataFrame(runs)
    
    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(sort_by)
    
    # Display columns
    display_cols = [
        "run_id", "model", "loss_type", "batch_size", "epochs",
        "duration_min", "precision", "gradient_clip", "aux_weight",
        "mae", "rmse", "r2"
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    df_display = df.fillna("—")
    
    print("="*200)
    print(f"TRAINING RUNS COMPARISON ({len(df)} runs, sorted by {sort_by})")
    print("="*200)
    print(df_display[display_cols].to_string(index=False))
    print("="*200)
    
    # Best performers
    print("\n🏆 TOP PERFORMERS:\n")
    
    for idx, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"{idx}. {row['run_id']}")
        print(f"   Model: {row['model']}, Loss: {row['loss_type'] if row['loss_type'] else 'N/A'}")
        print(f"   Epochs: {int(row['epochs']) if pd.notna(row['epochs']) else 'N/A'}, "
              f"Batch: {int(row['batch_size']) if pd.notna(row['batch_size']) else 'N/A'}")
        print(f"   MAE: {row['mae']:.6f}, RMSE: {row['rmse']:.6f}, R²: {row['r2']:.6f}\n")


def export_results(runs, format="csv"):
    """Export results to file."""
    if not runs:
        return
    
    df = pd.DataFrame(runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        filename = f"experiments/runs_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Exported to {filename}\n")
    elif format == "json":
        filename = f"experiments/runs_comparison_{timestamp}.json"
        df.to_json(filename, orient="records", indent=2)
        print(f"✓ Exported to {filename}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare all training runs")
    parser.add_argument("--sort-by", type=str, default="mae",
                        help="Column to sort by (default: mae)")
    parser.add_argument("--export", type=str, choices=["csv", "json"],
                        help="Export results to CSV or JSON")
    
    args = parser.parse_args()
    
    runs = extract_all_runs()
    print_comparison(runs, sort_by=args.sort_by)
    
    if args.export:
        export_results(runs, format=args.export)


if __name__ == "__main__":
    main()
def load_all_runs(experiments_dir="experiments"):
    """Load metadata from all completed runs and compute/extract metrics."""
    runs = []
    
    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return runs
    
    errors = []
    found_runs = 0
    checked_dirs = 0
    
    # Find all subdirectories with metadata.json (only direct children of experiments/)
    for run_dir in sorted(experiments_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        checked_dirs += 1
        metadata_file = run_dir / "metadata.json"
        
        # Only process if metadata.json exists in this directory
        if not metadata_file.exists():
            # Skip subdirectories without metadata.json
            continue
        
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Flatten metadata for easier comparison
            run_info = {
                "run_id": run_dir.name,
                "model": metadata.get("model_name"),
                "timestamp": metadata.get("timestamp"),
                "loss_type": metadata.get("loss_config", {}).get("loss_type"),
                "batch_size": metadata.get("training_config", {}).get("batch_size"),
                "epochs": metadata.get("training_config", {}).get("actual_epochs"),
                "duration_min": metadata.get("training_config", {}).get("training_duration_minutes"),
                "precision": metadata.get("precision_config", {}).get("precision"),
                "gradient_clip": metadata.get("loss_config", {}).get("gradient_clip_val"),
                "aux_weight": metadata.get("loss_config", {}).get("aux_weight"),
                "scaler_type": metadata.get("data_loader_config", {}).get("scaler_type"),
                "early_stop_patience": metadata.get("early_stopping_config", {}).get("patience"),
            }
            
            # Try to load metrics from metadata.json first
            metrics_loaded_from_file = False
            metrics = metadata.get("metrics", {})
            if metrics:
                for metric_type in ["overall_edc", "edt", "t20", "c50"]:
                    if metric_type in metrics:
                        metric_data = metrics[metric_type]
                        run_info[f"{metric_type}_mae"] = metric_data.get("mae")
                        run_info[f"{metric_type}_rmse"] = metric_data.get("rmse")
                        run_info[f"{metric_type}_r2"] = metric_data.get("r2")
                metrics_loaded_from_file = True
            
            # If metrics not in metadata, try to compute from predictions.npy and targets.npy
            if not metrics_loaded_from_file:
                preds_file = run_dir / "predictions.npy"
                targets_file = run_dir / "targets.npy"
                
                if preds_file.exists() and targets_file.exists():
                    try:
                        preds = np.load(preds_file)
                        targets = np.load(targets_file)
                        
                        # Compute metrics using the simple function
                        computed_metrics = compute_metrics_simple(targets, preds)
                        
                        # Add computed metrics to run_info
                        for metric_type in ["overall_edc", "edt", "t20", "c50"]:
                            if metric_type in computed_metrics:
                                metric_data = computed_metrics[metric_type]
                                run_info[f"{metric_type}_mae"] = metric_data.get("mae")
                                run_info[f"{metric_type}_rmse"] = metric_data.get("rmse")
                                run_info[f"{metric_type}_r2"] = metric_data.get("r2")
                    except Exception as e:
                        # Metrics computation failed, continue without them
                        pass
            
            runs.append(run_info)
            found_runs += 1
        
        except json.JSONDecodeError as e:
            errors.append(f"  ⚠ {run_dir.name}: JSON parsing error ({str(e)[:60]})")
        except Exception as e:
            errors.append(f"  ⚠ {run_dir.name}: {type(e).__name__} ({str(e)[:60]})")
    
    if errors:
        print("\n⚠ Skipped malformed runs:")
        for error in errors:
            print(error)
    
    print(f"\n✓ Checked {checked_dirs} directories, loaded {found_runs} valid runs")
    
    # DEBUG: Show what we actually loaded
    if found_runs > 0:
        print(f"   Loaded run IDs: {[r['run_id'] for r in runs]}\n")
    else:
        print("   WARNING: No runs found with complete metadata!\n")
    
    return runs


def print_summary(runs, sort_by="duration_min"):
    """Print a nicely formatted summary table."""
    if not runs:
        print("No runs found.")
        return
    
    print(f"\nCreating DataFrame with {len(runs)} runs...")
    
    df = pd.DataFrame(runs)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Unique run_ids: {df['run_id'].nunique()}")
    
    # Remove duplicate rows (in case of duplicate runs in the list)
    original_count = len(df)
    df = df.drop_duplicates(subset=['run_id'])
    if len(df) < original_count:
        print(f"⚠ Removed {original_count - len(df)} duplicate entries")
    
    print(f"After dedup: {len(df)} rows")
    print(f"DataFrame columns: {list(df.columns)}\n")
    
    # Fill None/NaN with "—" for display
    df_display = df.fillna("—").reset_index(drop=True)
    
    # Sort by specified metric (only if column exists and has numeric values)
    if sort_by in df.columns:
        # Try to sort by numeric values, ignoring non-numeric
        try:
            numeric_df = pd.to_numeric(df[sort_by], errors='coerce')
            # Only sort if there's at least one valid numeric value
            if numeric_df.notna().any():
                sort_indices = numeric_df.argsort(kind='stable')
                df_display = df_display.iloc[sort_indices].reset_index(drop=True)
        except Exception as e:
            # If sorting fails, just use original order
            pass
    
    # Display key columns (only those that exist)
    display_cols = [
        "run_id", "model", "loss_type", "batch_size", "epochs", 
        "duration_min", "precision", "gradient_clip", "aux_weight",
        "overall_edc_mae", "edt_mae", "t20_mae", "c50_mae"
    ]
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in df_display.columns]
    
    print("="*180)
    print(f"TRAINING RUNS COMPARISON ({len(df)} runs)")
    print("="*180)
    print(df_display[display_cols].to_string(index=False))
    print("="*180)
    
    # Print best performers for key metrics (only numeric values)
    print("\n🏆 BEST PERFORMERS:\n")
    metrics_to_track = [
        ("duration_min", "Fastest"),
        ("overall_edc_mae", "Best Overall EDC MAE"),
        ("edt_mae", "Best EDT MAE"),
        ("t20_mae", "Best T20 MAE"),
        ("c50_mae", "Best C50 MAE")
    ]
    
    any_metrics_found = False
    for metric, label in metrics_to_track:
        if metric in df.columns:
            # Convert to numeric, ignoring errors
            numeric_vals = pd.to_numeric(df[metric], errors='coerce')
            if numeric_vals.notna().any():  # If there's at least one numeric value
                any_metrics_found = True
                best_idx = numeric_vals.idxmin()
                best_val = numeric_vals[best_idx]
                if pd.notna(best_val):
                    run_id = df.loc[best_idx, 'run_id']
                    print(f"  {label:25s}: {run_id:35s} ({best_val:.4f})")
    
    if not any_metrics_found:
        print("  ℹ️  Metrics only available for runs created after 2026-01-21")
        print("     Showing all runs by epoch count instead:\n")
        # Show latest runs (which should have metrics)
        for idx, row in df.sort_values('epochs', ascending=False).head(5).iterrows():
            print(f"  - {row['run_id']}: {int(row['epochs'])} epochs, {row['model']}, {row['loss_type'] if row['loss_type'] != '—' else 'MSE'}")


def export_results(runs, format="csv"):
    """Export results to CSV or JSON."""
    df = pd.DataFrame(runs)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        filename = f"experiments/runs_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✓ Results exported to {filename}")
    elif format == "json":
        filename = f"experiments/runs_comparison_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"\n✓ Results exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Compare all training runs")
    parser.add_argument(
        "--sort-by", type=str, default="duration_min",
        help="Column to sort by (default: duration_min)"
    )
    parser.add_argument(
        "--export", type=str, choices=["csv", "json"],
        help="Export results to CSV or JSON"
    )
    
    args = parser.parse_args()
    
    runs = load_all_runs()
    print_summary(runs, sort_by=args.sort_by)
    
    if args.export:
        export_results(runs, format=args.export)


if __name__ == "__main__":
    main()
