"""
Evaluation metrics for EDC prediction.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple


def compute_acoustic_parameters(edc: np.ndarray, sample_rate: int = 48000) -> Dict[str, float]:
    """
    Compute acoustic parameters (EDT, T20, C50) from EDC.
    
    Args:
        edc: Energy Decay Curve (1D array)
        sample_rate: Sampling rate in Hz
        
    Returns:
        Dictionary containing EDT, T20, C50 values
    """
    # Normalize EDC to start at 0 dB
    if edc.max() <= 0:
        edc_db = edc.copy()
    else:
        edc_db = 10 * np.log10(np.maximum(edc, 1e-10) / edc.max())
    
    time_s = np.arange(len(edc)) / sample_rate
    
    # EDT: Time to decay from 0 to -10 dB
    idx_10db = np.where(edc_db <= -10)[0]
    edt = time_s[idx_10db[0]] if len(idx_10db) > 0 else np.nan
    
    # T20: Time to decay from -5 to -25 dB (extrapolated to 60 dB)
    idx_5db = np.where(edc_db <= -5)[0]
    idx_25db = np.where(edc_db <= -25)[0]
    
    if len(idx_5db) > 0 and len(idx_25db) > 0:
        time_5db = time_s[idx_5db[0]]
        time_25db = time_s[idx_25db[0]]
        t20 = 3 * (time_25db - time_5db)
    else:
        t20 = np.nan
    
    # C50: Clarity index (energy in first 50ms / total energy after 50ms)
    idx_50ms = np.argmin(np.abs(time_s - 0.05))
    early_energy = np.sum(edc[:idx_50ms])
    late_energy = np.sum(edc[idx_50ms:])
    
    if late_energy > 0:
        c50 = 10 * np.log10(early_energy / late_energy) if early_energy > 0 else -np.inf
    else:
        c50 = np.nan
    
    return {
        "edt": edt,
        "t20": t20,
        "c50": c50
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler_y = None,
    compute_acoustic: bool = True,
    sample_rate: int = 48000
) -> Dict[str, float]:
    """
    Evaluate model predictions.
    
    Args:
        y_true: Ground truth EDCs (batch, seq_len)
        y_pred: Predicted EDCs (batch, seq_len)
        scaler_y: Scaler to inverse transform (optional)
        compute_acoustic: Whether to compute acoustic parameters
        sample_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of metrics
    """
    # Inverse transform if scaler provided
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true)
        y_pred = scaler_y.inverse_transform(y_pred)
    
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics = {
        "mae": mean_absolute_error(y_true_flat, y_pred_flat),
        "rmse": np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        "r2": r2_score(y_true_flat, y_pred_flat),
    }
    
    # Compute per-sample acoustic metrics if requested
    if compute_acoustic:
        edts_true = []
        edts_pred = []
        t20s_true = []
        t20s_pred = []
        c50s_true = []
        c50s_pred = []
        
        for i in range(len(y_true)):
            # Ground truth
            params_true = compute_acoustic_parameters(y_true[i], sample_rate)
            edts_true.append(params_true["edt"])
            t20s_true.append(params_true["t20"])
            c50s_true.append(params_true["c50"])
            
            # Prediction
            params_pred = compute_acoustic_parameters(y_pred[i], sample_rate)
            edts_pred.append(params_pred["edt"])
            t20s_pred.append(params_pred["t20"])
            c50s_pred.append(params_pred["c50"])
        
        # Filter out NaN values
        edts_true = np.array([x for x in edts_true if not np.isnan(x)])
        edts_pred = np.array([x for x in edts_pred if not np.isnan(x)])
        t20s_true = np.array([x for x in t20s_true if not np.isnan(x)])
        t20s_pred = np.array([x for x in t20s_pred if not np.isnan(x)])
        c50s_true = np.array([x for x in c50s_true if not np.isnan(x)])
        c50s_pred = np.array([x for x in c50s_pred if not np.isnan(x)])
        
        # Add acoustic metrics
        if len(edts_true) > 0:
            metrics["edt_mae"] = mean_absolute_error(edts_true, edts_pred)
            metrics["edt_rmse"] = np.sqrt(mean_squared_error(edts_true, edts_pred))
            metrics["edt_r2"] = r2_score(edts_true, edts_pred)
        
        if len(t20s_true) > 0:
            metrics["t20_mae"] = mean_absolute_error(t20s_true, t20s_pred)
            metrics["t20_rmse"] = np.sqrt(mean_squared_error(t20s_true, t20s_pred))
            metrics["t20_r2"] = r2_score(t20s_true, t20s_pred)
        
        if len(c50s_true) > 0:
            metrics["c50_mae"] = mean_absolute_error(c50s_true, c50s_pred)
            metrics["c50_rmse"] = np.sqrt(mean_squared_error(c50s_true, c50s_pred))
            metrics["c50_r2"] = r2_score(c50s_true, c50s_pred)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a readable format."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Overall metrics
    print("\nOverall EDC Metrics:")
    print(f"  MAE:  {metrics.get('mae', np.nan):.6f}")
    print(f"  RMSE: {metrics.get('rmse', np.nan):.6f}")
    print(f"  R²:   {metrics.get('r2', np.nan):.6f}")
    
    # Acoustic metrics
    if "edt_mae" in metrics:
        print("\nEDT Metrics:")
        print(f"  MAE:  {metrics['edt_mae']:.6f}")
        print(f"  RMSE: {metrics['edt_rmse']:.6f}")
        print(f"  R²:   {metrics['edt_r2']:.6f}")
    
    if "t20_mae" in metrics:
        print("\nT20 Metrics:")
        print(f"  MAE:  {metrics['t20_mae']:.6f}")
        print(f"  RMSE: {metrics['t20_rmse']:.6f}")
        print(f"  R²:   {metrics['t20_r2']:.6f}")
    
    if "c50_mae" in metrics:
        print("\nC50 Metrics:")
        print(f"  MAE:  {metrics['c50_mae']:.6f}")
        print(f"  RMSE: {metrics['c50_rmse']:.6f}")
        print(f"  R²:   {metrics['c50_r2']:.6f}")
    
    print("\n" + "="*50)
