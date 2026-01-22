"""
Compare all training runs and generate a summary report.

Usage:
    python scripts/compare_runs.py [--sort-by metric] [--export csv/json]

Reads metadata.json from all experiment directories and produces a comparison table.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_all_runs(experiments_dir="experiments"):
    """Load metadata from all completed runs."""
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
            
            # Add metrics (flatten nested dict)
            metrics = metadata.get("metrics", {})
            for metric_type in ["overall_edc", "edt", "t20", "c50"]:
                if metric_type in metrics:
                    metric_data = metrics[metric_type]
                    run_info[f"{metric_type}_mae"] = metric_data.get("mae")
                    run_info[f"{metric_type}_rmse"] = metric_data.get("rmse")
                    run_info[f"{metric_type}_r2"] = metric_data.get("r2")
            
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
    
    print(f"After dedup: {len(df)} rows\n")
    
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
    
    for metric, label in metrics_to_track:
        if metric in df.columns:
            # Convert to numeric, ignoring errors
            numeric_vals = pd.to_numeric(df[metric], errors='coerce')
            if numeric_vals.notna().any():  # If there's at least one numeric value
                best_idx = numeric_vals.idxmin()
                best_val = numeric_vals[best_idx]
                if pd.notna(best_val):
                    run_id = df.loc[best_idx, 'run_id']
                    print(f"  {label:25s}: {run_id:35s} ({best_val:.4f})")


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
