#!/usr/bin/env python3
"""
Generate publication-ready plots for multihead EDC model results.
Matches reference paper style: decay curves, scatter plots, error distributions.

Usage:
  python scripts/plot_results.py --run-dir experiments/multihead_20260123_120009
"""
from pathlib import Path
import argparse
import sys
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_acoustic_parameters  # noqa: E402

# Publication-style configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def edc_to_db(edc: np.ndarray) -> np.ndarray:
    """Convert linear EDC to dB scale."""
    peak = np.max(edc)
    safe = np.maximum(edc, 1e-10)
    return 10 * np.log10(safe / peak) if peak > 0 else 10 * np.log10(safe)


def compute_edt_batch(edc_batch: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Compute EDT for batch of EDCs."""
    values = []
    for edc in edc_batch:
        params = compute_acoustic_parameters(edc, sample_rate=sample_rate)
        values.append(params["edt"])
    return np.array(values)


def pick_indices_by_error(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> list:
    """Pick low/median/high error samples."""
    errors = np.abs(y_true - y_pred)
    if n == 1:
        return [np.argmin(errors)]
    if n == 2:
        return [np.argmin(errors), np.argmax(errors)]
    sorted_idx = np.argsort(errors)
    low = sorted_idx[0]
    high = sorted_idx[-1]
    med = sorted_idx[len(sorted_idx) // 2]
    return [low, med, high]


def load_arrays(run_dir):
    """Load prediction and target arrays."""
    arrays = {}
    for name in [
        "edc_predictions.npy",
        "edc_targets.npy",
        "t20_predictions.npy",
        "t20_targets.npy",
        "c50_predictions.npy",
        "c50_targets.npy",
    ]:
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        arrays[name] = np.load(path)
    return arrays


def compute_metrics(t_true, t_pred):
    """Compute MAE, RMSE, R² metrics."""
    return {
        "mae": mean_absolute_error(t_true, t_pred),
        "rmse": np.sqrt(mean_squared_error(t_true, t_pred)),
        "r2": r2_score(t_true, t_pred),
    }


def plot_scatter_with_stats(ax, y_true, y_pred, title: str, color: str, 
                            show_histogram: bool = False) -> None:
    """Scatter plot with 1:1 line and embedded metrics."""
    ax.scatter(y_true, y_pred, s=20, alpha=0.5, color=color, edgecolors='none')
    
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max], 'k-', linewidth=1.5, label='Perfect prediction')
    
    metrics = compute_metrics(y_true, y_pred)
    
    ax.set_xlabel('Target value', fontsize=10)
    ax.set_ylabel('Predicted value', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Embed metrics in text box
    metrics_text = (
        f"MAE: {metrics['mae']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"R²: {metrics['r2']:.4f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return metrics


def plot_decay_curves_with_regression(ax, edc_true: np.ndarray, edc_pred: np.ndarray, 
                                       sample_rate: int = 48000, max_samples: int = 3) -> None:
    """Plot EDC decay curves in dB with regression lines for EDT/T20."""
    idxs = pick_indices_by_error(edc_true[:, 0], edc_pred[:, 0], n=max_samples)
    
    colors_true = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_pred = ['#aec7e8', '#ffbb78', '#98df8a']
    
    time_s = np.arange(edc_true.shape[1]) / sample_rate
    max_time = time_s[-1]
    
    for plot_idx, sample_idx in enumerate(idxs):
        edc_t = edc_true[sample_idx]
        edc_p = edc_pred[sample_idx]
        
        edc_t_db = edc_to_db(edc_t)
        edc_p_db = edc_to_db(edc_p)
        
        label_t = f"Target (sample {sample_idx})"
        label_p = f"Predicted (sample {sample_idx})"
        
        ax.plot(time_s, edc_t_db, linestyle='-', linewidth=1.5, 
                color=colors_true[plot_idx], label=label_t, alpha=0.8)
        ax.plot(time_s, edc_p_db, linestyle='--', linewidth=1.5, 
                color=colors_pred[plot_idx], label=label_p, alpha=0.8)
        
        # Mark EDT (-10 dB) and T20 (-5 to -25 dB) regions
        edt_idx_t = np.where(edc_t_db <= -10)[0]
        if len(edt_idx_t) > 0:
            edt_time_t = time_s[edt_idx_t[0]]
            ax.plot(edt_time_t, -10, 'o', color=colors_true[plot_idx], markersize=5, alpha=0.6)
    
    ax.axhline(-10, color='gray', linestyle=':', linewidth=0.8, alpha=0.4, label='EDT (-10 dB)')
    ax.axhline(-5, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.axhline(-25, color='gray', linestyle=':', linewidth=0.8, alpha=0.4, label='T20 (-5 to -25 dB)')
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('EDC (dB)', fontsize=10)
    ax.set_title('Energy Decay Curves (dB) - Sample Comparison', fontsize=11, fontweight='bold')
    ax.set_xlim([0, max_time])
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)


def plot_error_distribution(ax, errors: np.ndarray, title: str, color: str) -> None:
    """Plot histogram of absolute prediction errors."""
    ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
    ax.set_xlabel('Absolute Error', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, axis='y')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready plots for multihead EDC model")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: overview_plots.png)")
    args = parser.parse_args()

    # Locate run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        experiments_dir = ROOT / "experiments"
        if not experiments_dir.exists():
            raise SystemExit("experiments/ directory not found")
        runs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()], 
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise SystemExit("No experiments found")
        run_dir = runs[0]

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    print(f"Loading data from {run_dir.name}...")
    arrays = load_arrays(run_dir)
    edc_pred = arrays["edc_predictions.npy"]
    edc_true = arrays["edc_targets.npy"]
    t20_pred = arrays["t20_predictions.npy"]
    t20_true = arrays["t20_targets.npy"]
    c50_pred = arrays["c50_predictions.npy"]
    c50_true = arrays["c50_targets.npy"]

    # Compute EDT for batch
    edt_true = compute_edt_batch(edc_true)
    edt_pred = compute_edt_batch(edc_pred)

    # Compute errors for histograms
    t20_errors = np.abs(t20_true - t20_pred)
    c50_errors = np.abs(c50_true - c50_pred)
    edt_errors = np.abs(edt_true - edt_pred)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Row 1: Scatter plots for T20, C50, EDT
    ax_t20_scatter = fig.add_subplot(gs[0, 0])
    ax_c50_scatter = fig.add_subplot(gs[0, 1])
    ax_edt_scatter = fig.add_subplot(gs[0, 2])

    plot_scatter_with_stats(ax_t20_scatter, t20_true, t20_pred, "T20 (Reverberation Time)", "#1f77b4")
    plot_scatter_with_stats(ax_c50_scatter, c50_true, c50_pred, "C50 (Clarity Index)", "#2ca02c")
    plot_scatter_with_stats(ax_edt_scatter, edt_true, edt_pred, "EDT (Early Decay Time)", "#ff7f0e")

    # Row 2: Error distributions
    ax_t20_hist = fig.add_subplot(gs[1, 0])
    ax_c50_hist = fig.add_subplot(gs[1, 1])
    ax_edt_hist = fig.add_subplot(gs[1, 2])

    plot_error_distribution(ax_t20_hist, t20_errors, "T20 Error Distribution", "#1f77b4")
    plot_error_distribution(ax_c50_hist, c50_errors, "C50 Error Distribution", "#2ca02c")
    plot_error_distribution(ax_edt_hist, edt_errors, "EDT Error Distribution", "#ff7f0e")

    # Row 3: EDC decay curves with regression
    ax_edc = fig.add_subplot(gs[2, :])
    plot_decay_curves_with_regression(ax_edc, edc_true, edc_pred, max_samples=3)

    fig.suptitle(f"Multi-Head EDC Prediction Model - {run_dir.name}", 
                 fontsize=14, fontweight='bold', y=0.995)

    # Save output
    output_file = args.output if args.output else run_dir / "overview_plots.png"
    output_path = Path(output_file)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved publication-ready plots to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nT20 (Reverberation Time)")
    print(f"  MAE:  {mean_absolute_error(t20_true, t20_pred):.6f} s")
    print(f"  RMSE: {np.sqrt(mean_squared_error(t20_true, t20_pred)):.6f} s")
    print(f"  R²:   {r2_score(t20_true, t20_pred):.6f}")
    
    print(f"\nC50 (Clarity Index)")
    print(f"  MAE:  {mean_absolute_error(c50_true, c50_pred):.6f} dB")
    print(f"  RMSE: {np.sqrt(mean_squared_error(c50_true, c50_pred)):.6f} dB")
    print(f"  R²:   {r2_score(c50_true, c50_pred):.6f}")
    
    print(f"\nEDT (Early Decay Time)")
    print(f"  MAE:  {mean_absolute_error(edt_true, edt_pred):.6f} s")
    print(f"  RMSE: {np.sqrt(mean_squared_error(edt_true, edt_pred)):.6f} s")
    print(f"  R²:   {r2_score(edt_true, edt_pred):.6f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
