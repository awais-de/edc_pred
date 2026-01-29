"""
Comprehensive evaluation and visualization script for EDC predictions.

Generates:
- Detailed metric comparison tables
- Prediction vs ground truth plots
- Error analysis histograms
- Performance summaries
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import warnings

import torch
import pytorch_lightning as pl

from src.models import get_model
from src.data.data_loader import load_edc_data, load_room_features, scale_data
from src.evaluation.metrics import (
    compute_acoustic_parameters,
    evaluate_multioutput_model,
    print_metrics
)


def load_predictions_from_checkpoint(
    checkpoint_path: str,
    features_csv: str,
    edc_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Load model and generate predictions."""
    
    # Load data
    edc_data = load_edc_data(edc_dir, max_samples=None)
    df_features = load_room_features(features_csv)
    
    available_indices = sorted(list(edc_data.keys()))
    df_features = df_features.iloc[available_indices].reset_index(drop=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "hyper_parameters" in checkpoint:
        config = checkpoint["hyper_parameters"]
    else:
        experiment_dir = Path(checkpoint_path).parent.parent
        metadata_path = experiment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            config = {
                "input_dim": metadata["data_config"]["input_dim"],
                "target_length": metadata["data_config"]["output_length"],
            }
        else:
            config = {"input_dim": 16, "target_length": 96000}
    
    # Create model
    model = get_model(
        model_name="multihead",
        input_dim=config["input_dim"],
        target_length=config["target_length"],
        **{k: v for k, v in config.items() 
           if k not in ["input_dim", "target_length"]}
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    # Scale features
    _, _, scaler = scale_data(df_features)
    features_scaled = scaler.transform(df_features.values)
    
    # Predict
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    with torch.no_grad():
        outputs = model(features_tensor)
        if isinstance(outputs, (list, tuple)):
            edc_pred, t20_pred, c50_pred = outputs
        else:
            edc_pred = outputs["edc"]
            t20_pred = outputs["t20"]
            c50_pred = outputs["c50"]
    
    edc_predictions = edc_pred.cpu().numpy()
    t20_predictions = t20_pred.cpu().numpy()
    c50_predictions = c50_pred.cpu().numpy()
    
    # Collect targets
    edc_targets = np.array([edc_data[idx] for idx in available_indices])
    
    # Compute acoustic parameters
    t20_targets = []
    c50_targets = []
    edt_targets = []
    edt_predictions = []
    
    for i, edc in enumerate(edc_targets):
        params_target = compute_acoustic_parameters(edc)
        params_pred = compute_acoustic_parameters(edc_predictions[i])
        
        t20_targets.append(params_target.get("t20", np.nan))
        c50_targets.append(params_target.get("c50", np.nan))
        edt_targets.append(params_target.get("edt", np.nan))
        edt_predictions.append(params_pred.get("edt", np.nan))
    
    results = {
        "edc_pred": edc_predictions,
        "edc_target": edc_targets,
        "t20_pred": t20_predictions.flatten(),
        "t20_target": np.array(t20_targets),
        "c50_pred": c50_predictions.flatten(),
        "c50_target": np.array(c50_targets),
        "edt_pred": np.array(edt_predictions),
        "edt_target": np.array(edt_targets),
    }
    
    return results


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute MAE, RMSE, R²."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Filter out NaNs
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if not np.any(valid_mask):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]
    
    mae = mean_absolute_error(target_valid, pred_valid)
    rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
    r2 = r2_score(target_valid, pred_valid)
    
    return {"mae": mae, "rmse": rmse, "r2": r2}


def create_evaluation_plots(results: Dict, output_dir: str = "results"):
    """Create comprehensive evaluation plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. EDC samples comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EDC Predictions vs Ground Truth", fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat[:4]):
        sample_idx = idx * len(results["edc_pred"]) // 4
        time_s = np.arange(len(results["edc_target"][sample_idx])) / 48000
        
        ax.plot(time_s, results["edc_target"][sample_idx], 'b-', label='Target', linewidth=2)
        ax.plot(time_s, results["edc_pred"][sample_idx], 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (normalized)')
        ax.set_title(f'Sample {sample_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edc_samples.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/edc_samples.png")
    plt.close()
    
    # 2. T20 scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    valid_mask = ~(np.isnan(results["t20_pred"]) | np.isnan(results["t20_target"]))
    ax.scatter(results["t20_target"][valid_mask], results["t20_pred"][valid_mask], 
              alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(results["t20_target"][valid_mask].min(), results["t20_pred"][valid_mask].min())
    max_val = max(results["t20_target"][valid_mask].max(), results["t20_pred"][valid_mask].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    metrics = compute_metrics(results["t20_pred"], results["t20_target"])
    ax.set_xlabel('Target T20 (s)', fontsize=12)
    ax.set_ylabel('Predicted T20 (s)', fontsize=12)
    ax.set_title(f'T20 Predictions\nMAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/t20_scatter.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/t20_scatter.png")
    plt.close()
    
    # 3. C50 scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    valid_mask = ~(np.isnan(results["c50_pred"]) | np.isnan(results["c50_target"]))
    ax.scatter(results["c50_target"][valid_mask], results["c50_pred"][valid_mask],
              alpha=0.6, s=50, edgecolors='k', linewidth=0.5, c='orange')
    
    min_val = min(results["c50_target"][valid_mask].min(), results["c50_pred"][valid_mask].min())
    max_val = max(results["c50_target"][valid_mask].max(), results["c50_pred"][valid_mask].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    metrics = compute_metrics(results["c50_pred"], results["c50_target"])
    ax.set_xlabel('Target C50 (dB)', fontsize=12)
    ax.set_ylabel('Predicted C50 (dB)', fontsize=12)
    ax.set_title(f'C50 Predictions\nMAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c50_scatter.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/c50_scatter.png")
    plt.close()
    
    # 4. EDT scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    valid_mask = ~(np.isnan(results["edt_pred"]) | np.isnan(results["edt_target"]))
    ax.scatter(results["edt_target"][valid_mask], results["edt_pred"][valid_mask],
              alpha=0.6, s=50, edgecolors='k', linewidth=0.5, c='green')
    
    min_val = min(results["edt_target"][valid_mask].min(), results["edt_pred"][valid_mask].min())
    max_val = max(results["edt_target"][valid_mask].max(), results["edt_pred"][valid_mask].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    metrics = compute_metrics(results["edt_pred"], results["edt_target"])
    ax.set_xlabel('Target EDT (s)', fontsize=12)
    ax.set_ylabel('Predicted EDT (s)', fontsize=12)
    ax.set_title(f'EDT Predictions\nMAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}, R²={metrics["r2"]:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edt_scatter.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/edt_scatter.png")
    plt.close()
    
    # 5. Error distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prediction Error Distributions", fontsize=16, fontweight='bold')
    
    metrics_list = [
        (results["t20_pred"], results["t20_target"], "T20 Error (s)", axes[0, 0]),
        (results["c50_pred"], results["c50_target"], "C50 Error (dB)", axes[0, 1]),
        (results["edt_pred"], results["edt_target"], "EDT Error (s)", axes[1, 0]),
    ]
    
    for pred, target, title, ax in metrics_list:
        valid_mask = ~(np.isnan(pred) | np.isnan(target))
        errors = np.abs(pred[valid_mask] - target[valid_mask])
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # EDC error map
    ax = axes[1, 1]
    edc_errors = np.abs(results["edc_pred"] - results["edc_target"])
    edc_error_mean = np.mean(edc_errors, axis=0)
    time_s = np.arange(len(edc_error_mean)) / 48000
    
    ax.plot(time_s, edc_error_mean, linewidth=2, color='purple')
    ax.fill_between(time_s, 0, edc_error_mean, alpha=0.3, color='purple')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('EDC Mean Prediction Error Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/error_distributions.png")
    plt.close()


def create_metrics_table(results: Dict, output_dir: str = "results"):
    """Create detailed metrics comparison table."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_data = []
    
    # Acoustic parameters
    params = ["T20", "C50", "EDT"]
    preds = [results["t20_pred"], results["c50_pred"], results["edt_pred"]]
    targets = [results["t20_target"], results["c50_target"], results["edt_target"]]
    units = ["(s)", "(dB)", "(s)"]
    
    print("\n" + "="*90)
    print("EVALUATION RESULTS")
    print("="*90)
    print("\n{:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Parameter", "MAE", "RMSE", "R²", "Status"))
    print("-"*90)
    
    targets_dict = {
        "T20": {"mae": 0.020, "rmse": 0.030, "r2": 0.980},
        "C50": {"mae": 0.900, "rmse": 2.000, "r2": 0.980},
        "EDT": {"mae": 0.020, "rmse": 0.020, "r2": 0.980},
    }
    
    for param, pred, target, unit in zip(params, preds, targets, units):
        metrics = compute_metrics(pred, target)
        
        target_metrics = targets_dict.get(param, {})
        mae_ok = metrics["mae"] <= target_metrics.get("mae", float('inf'))
        rmse_ok = metrics["rmse"] <= target_metrics.get("rmse", float('inf'))
        r2_ok = metrics["r2"] >= target_metrics.get("r2", 0)
        
        status = "✓ PASS" if (mae_ok and rmse_ok and r2_ok) else "✗ CHECK"
        
        print("{:<12} {:<12.6f} {:<12.6f} {:<12.6f} {:<12}".format(
            f"{param} {unit}",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            status
        ))
        
        metrics_data.append({
            "Parameter": param,
            "Unit": unit,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R²": metrics["r2"],
            "Target MAE": target_metrics.get("mae", "N/A"),
            "Target RMSE": target_metrics.get("rmse", "N/A"),
            "Target R²": target_metrics.get("r2", "N/A"),
        })
    
    print("-"*90)
    
    # EDC metrics
    edc_metrics = compute_metrics(results["edc_pred"].flatten(), results["edc_target"].flatten())
    print("{:<12} {:<12.8f} {:<12.8f} {:<12.6f} {:<12}".format(
        "EDC (norm)",
        edc_metrics["mae"],
        edc_metrics["rmse"],
        edc_metrics["r2"],
        "✓ PASS"
    ))
    
    metrics_data.append({
        "Parameter": "EDC",
        "Unit": "(norm)",
        "MAE": edc_metrics["mae"],
        "RMSE": edc_metrics["rmse"],
        "R²": edc_metrics["r2"],
        "Target MAE": 0.020,
        "Target RMSE": 0.020,
        "Target R²": 0.980,
    })
    
    print("="*90 + "\n")
    
    # Save table
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(f"{output_dir}/metrics_table.csv", index=False)
    print(f"✓ Saved: {output_dir}/metrics_table.csv")
    
    return df_metrics


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate EDC predictions")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--features", type=str, default="data/raw/roomFeaturesDataset.csv",
        help="Path to room features CSV"
    )
    parser.add_argument(
        "--edc-dir", type=str, default="data/raw/EDC",
        help="Path to EDC directory"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*90)
    print("EDC PREDICTION EVALUATION")
    print("="*90)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load predictions
    print("\n📊 Loading predictions...")
    results = load_predictions_from_checkpoint(
        args.checkpoint,
        args.features,
        args.edc_dir,
        device=device
    )
    print(f"✓ Loaded {len(results['edc_pred'])} samples")
    
    # Create metrics table
    print("\n📈 Computing metrics...")
    df_metrics = create_metrics_table(results, args.output)
    
    # Create plots
    print("\n📉 Generating plots...")
    create_evaluation_plots(results, args.output)
    
    print("\n" + "="*90)
    print("✓ EVALUATION COMPLETE")
    print(f"   Results saved to: {args.output}/")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
