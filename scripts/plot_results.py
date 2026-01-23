#!/usr/bin/env python3
"""
Generate summary plots for a run: T20/C50/EDT scatter and sample EDC overlays.
Usage:
  python scripts/plot_results.py --run-dir experiments/multihead_20260123_120009
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_acoustic_parameters  # noqa: E402


def edc_to_db(edc: np.ndarray) -> np.ndarray:
    peak = np.max(edc)
    safe = np.maximum(edc, 1e-10)
    return 10 * np.log10(safe / peak) if peak > 0 else 10 * np.log10(safe)


def compute_edt_batch(edc_batch: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    values = []
    for edc in edc_batch:
        params = compute_acoustic_parameters(edc, sample_rate=sample_rate)
        values.append(params["edt"])
    return np.array(values)


def pick_indices(n: int) -> list[int]:
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    mid = n // 2
    return [0, mid, n - 1]


def load_arrays(run_dir: Path) -> dict[str, np.ndarray]:
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


def compute_summary_metrics(t_true, t_pred, label: str) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(t_true, t_pred),
        "rmse": np.sqrt(mean_squared_error(t_true, t_pred)),
        "r2": r2_score(t_true, t_pred),
        "label": label,
    }


def add_scatter(ax, y_true, y_pred, metrics: dict[str, float], title: str, color: str) -> None:
    ax.scatter(y_true, y_pred, s=6, alpha=0.6, color=color, edgecolors="none")
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max], "k--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(
        0.02,
        0.98,
        f"MAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nR2={metrics['r2']:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=9,
    )


def add_edc_overlays(ax, edc_true, edc_pred, sample_rate: int = 48000) -> None:
    idxs = pick_indices(len(edc_true))
    for i, idx in enumerate(idxs):
        t = np.arange(edc_true.shape[1]) / sample_rate
        ax.plot(t, edc_to_db(edc_true[idx]), label=f"Target #{idx}", linestyle="-", linewidth=1.2)
        ax.plot(t, edc_to_db(edc_pred[idx]), label=f"Pred #{idx}", linestyle="--", linewidth=1.2)
        if i >= 2:
            break
    ax.set_title("EDC overlays (dB vs time)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EDC (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multihead run results")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to run directory under experiments/")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs = sorted([d for d in (ROOT / "experiments").iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise SystemExit("No experiments found")
        run_dir = runs[0]

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    arrays = load_arrays(run_dir)
    edc_pred = arrays["edc_predictions.npy"]
    edc_true = arrays["edc_targets.npy"]
    t20_pred = arrays["t20_predictions.npy"]
    t20_true = arrays["t20_targets.npy"]
    c50_pred = arrays["c50_predictions.npy"]
    c50_true = arrays["c50_targets.npy"]

    edt_true = compute_edt_batch(edc_true)
    edt_pred = compute_edt_batch(edc_pred)

    t20_metrics = compute_summary_metrics(t20_true, t20_pred, "T20")
    c50_metrics = compute_summary_metrics(c50_true, c50_pred, "C50")
    edt_metrics = compute_summary_metrics(edt_true, edt_pred, "EDT")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    add_scatter(axes[0, 0], t20_true, t20_pred, t20_metrics, "T20 scatter", "#C0A4FD")
    add_scatter(axes[0, 1], c50_true, c50_pred, c50_metrics, "C50 scatter", "#7CCBA2")
    add_scatter(axes[1, 0], edt_true, edt_pred, edt_metrics, "EDT scatter", "#F4A259")
    add_edc_overlays(axes[1, 1], edc_true, edc_pred)

    fig.suptitle(f"Run: {run_dir.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = run_dir / "overview_plots.png"
    fig.savefig(output_path, dpi=200)
    print(f"✓ Saved plots to {output_path}")


if __name__ == "__main__":
    main()
