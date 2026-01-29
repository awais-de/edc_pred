"""
Inference module for EDC prediction.

This script provides a complete inference pipeline for:
1. Loading trained models
2. Making predictions on new room configurations
3. Computing acoustic parameters (EDT, T20, C50)
4. Visualization and analysis
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from src.models import get_model
from src.data.data_loader import load_room_features, scale_data
from src.evaluation.metrics import compute_acoustic_parameters, evaluate_multioutput_model


class EDCPredictor:
    """
    High-level interface for EDC prediction inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        features_csv: str = "data/raw/roomFeaturesDataset.csv",
        device: str = "auto"
    ):
        """
        Initialize EDC predictor with a trained model.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            features_csv: Path to room features CSV for normalization
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load scaler from checkpoint directory (parent of checkpoints folder)
        checkpoint_dir = Path(checkpoint_path).parent.parent
        scaler_X_path = checkpoint_dir / "scaler_X.pkl"
        
        if not scaler_X_path.exists():
            print(f"⚠️  Scaler not found at {scaler_X_path}")
            print("   Will attempt to create scaler from features CSV")
            self.scaler = self._load_or_create_scaler(features_csv)
        else:
            import joblib
            self.scaler = joblib.load(scaler_X_path)
            print(f"✓ Loaded input scaler (scaler_X) from training")
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
    
    def _load_or_create_scaler(self, features_csv: str):
        """Load or create feature scaler."""
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        df = pd.read_csv(features_csv)
        # Exclude non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols].values)
        return scaler
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model config from checkpoint
        config = {}
        if "hyper_parameters" in checkpoint:
            config = checkpoint["hyper_parameters"].copy()
        
        # Ensure required parameters
        if not config.get("input_dim"):
            experiment_dir = Path(self.checkpoint_path).parent.parent
            metadata_path = experiment_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                config["input_dim"] = metadata.get("data_config", {}).get("input_dim", 16)
                config["target_length"] = metadata.get("data_config", {}).get("output_length", 96000)
        
        # Set defaults for missing keys
        config.setdefault("input_dim", 16)
        config.setdefault("target_length", 96000)
        config.setdefault("learning_rate", 0.001)
        config.setdefault("cnn_filters", [32, 64])
        config.setdefault("cnn_kernel_sizes", [3, 3])
        config.setdefault("lstm_hidden_dim", 128)
        config.setdefault("fc_hidden_dim", 2048)
        config.setdefault("dropout_rate", 0.3)
        config.setdefault("edc_weight", 1.0)
        config.setdefault("t20_weight", 1.0)
        config.setdefault("c50_weight", 1.0)
        
        cnn_filters_to_use = config.get("cnn_filters") or [32, 64]
        
        # Instantiate model with full config
        model = get_model(
            model_name="multihead",
            input_dim=config["input_dim"],
            target_length=config["target_length"],
            cnn_filters=cnn_filters_to_use,
            cnn_kernel_sizes=config.get("cnn_kernel_sizes") or [3, 3],
            lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
            fc_hidden_dim=config.get("fc_hidden_dim", 2048),
            dropout_rate=config.get("dropout_rate", 0.3),
            learning_rate=config["learning_rate"],
            edc_weight=config.get("edc_weight", 1.0),
            t20_weight=config.get("t20_weight", 1.0),
            c50_weight=config.get("c50_weight", 1.0),
        )
        
        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def predict(
        self,
        room_features: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on room features.
        
        Args:
            room_features: Array of shape (n_samples, n_features)
            normalize: Whether to normalize features using loaded scaler
            
        Returns:
            Dictionary with predictions for EDC, T20, C50
        """
        # Ensure 2D input
        if room_features.ndim == 1:
            room_features = room_features.reshape(1, -1)
        
        # Normalize if needed
        if normalize:
            room_features_scaled = self.scaler.transform(room_features)
        else:
            room_features_scaled = room_features
        
        # Convert to tensor - reshape for CNN input (batch_size, 1, input_dim)
        features_tensor = torch.FloatTensor(room_features_scaled).to(self.device)
        features_tensor = features_tensor.unsqueeze(1)  # Add channel dimension
        
        # Debug: Check model state before forward
        import sys
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            if isinstance(outputs, (list, tuple)):
                edc_pred, t20_pred, c50_pred = outputs
            else:
                # Handle case where model returns dict
                edc_pred = outputs.get("edc")
                t20_pred = outputs.get("t20")
                c50_pred = outputs.get("c50")
        
        return {
            "edc": edc_pred.cpu().numpy(),
            "t20": t20_pred.cpu().numpy(),
            "c50": c50_pred.cpu().numpy(),
        }
    
    def predict_and_analyze(
        self,
        room_features: np.ndarray,
        normalize: bool = True,
        sample_rate: int = 48000
    ) -> Dict:
        """
        Make predictions and compute acoustic parameters.
        
        Args:
            room_features: Array of shape (n_samples, n_features)
            normalize: Whether to normalize features
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with EDC and acoustic parameters
        """
        predictions = self.predict(room_features, normalize)
        
        results = {
            "edc_predictions": predictions["edc"],
            "t20_predictions": predictions["t20"],
            "c50_predictions": predictions["c50"],
            "acoustic_parameters": []
        }
        
        # Compute acoustic parameters for each EDC
        for edc in predictions["edc"]:
            params = compute_acoustic_parameters(edc, sample_rate)
            results["acoustic_parameters"].append(params)
        
        return results
    
    def plot_edc_curve(self, edc_data: np.ndarray, room_index: int = 0, 
                       sample_rate: int = 48000, save_path: Optional[str] = None):
        """
        Plot Energy Decay Curve.
        
        Args:
            edc_data: EDC array of shape (n_samples, 96000)
            room_index: Which room to plot (default 0)
            sample_rate: Sample rate in Hz
            save_path: Optional path to save the figure
        """
        edc = edc_data[room_index] if len(edc_data.shape) > 1 else edc_data
        
        # Convert samples to time (in seconds)
        time_samples = np.arange(len(edc))
        time_seconds = time_samples / sample_rate
        
        # Convert EDC to dB scale (normalized to max)
        edc_db = 10 * np.log10(np.maximum(edc, 1e-10))
        edc_db_normalized = edc_db - edc_db[0]  # Normalize to start at 0 dB
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Full EDC curve
        ax1.plot(time_seconds, edc_db_normalized, linewidth=1.5, color='#1f77b4')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Energy Decay (dB)', fontsize=11)
        ax1.set_title(f'Full Energy Decay Curve (EDC) - Room {room_index}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([edc_db_normalized.min() - 5, 5])
        
        # Early decay (first 100ms for better visibility)
        early_samples = int(0.1 * sample_rate)  # 100ms
        ax2.plot(time_seconds[:early_samples], edc_db_normalized[:early_samples], 
                linewidth=1.5, color='#ff7f0e', label='EDC')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Energy Decay (dB)', fontsize=11)
        ax2.set_title('Early Decay (First 100ms)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📈 EDC plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description="EDC Prediction Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict for a single room
  python inference.py --checkpoint trained_models/multihead_*/checkpoints/best_model.ckpt \\
                      --features data/raw/roomFeaturesDataset.csv --index 0
  
  # Predict for multiple rooms
  python inference.py --checkpoint trained_models/multihead_*/checkpoints/best_model.ckpt \\
                      --features data/raw/roomFeaturesDataset.csv --indices 0 1 2 3
  
  # Evaluate on test set
  python inference.py --checkpoint trained_models/multihead_*/checkpoints/best_model.ckpt \\
                      --features data/raw/roomFeaturesDataset.csv \\
                      --edc-dir data/raw/EDC --evaluate
        """
    )
    
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
        help="Path to EDC directory (for evaluation)"
    )
    parser.add_argument(
        "--index", type=int, nargs="?",
        help="Index of single room to predict"
    )
    parser.add_argument(
        "--indices", type=int, nargs="+",
        help="Indices of multiple rooms to predict"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run full evaluation on test set"
    )
    parser.add_argument(
        "--output", type=str, default="inference_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Sample rate in Hz"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("\n" + "="*60)
    print("EDC PREDICTION INFERENCE")
    print("="*60)
    
    predictor = EDCPredictor(args.checkpoint, args.features, args.device)
    
    # Load features
    df_features = pd.read_csv(args.features)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    if args.evaluate:
        print("\n📊 Running evaluation on test set...")
        evaluate_from_checkpoint(
            args.checkpoint,
            args.features,
            args.edc_dir,
            numeric_cols
        )
    
    elif args.index is not None:
        print(f"\n🔍 Predicting for room index {args.index}...")
        features = df_features.iloc[args.index][numeric_cols].values.reshape(1, -1)
        results = predictor.predict_and_analyze(features)
        
        print(f"\n✓ Prediction successful!")
        print(f"  EDC shape: {results['edc_predictions'].shape}")
        print(f"  T20: {results['t20_predictions'][0]:.4f} s")
        print(f"  C50: {results['c50_predictions'][0]:.4f} dB")
        if results['acoustic_parameters']:
            params = results['acoustic_parameters'][0]
            print(f"  EDT: {params.get('edt', np.nan):.4f} s")
        
        # Plot EDC curve if requested
        if args.visualize:
            plots_dir = Path("edc_plots")
            plots_dir.mkdir(exist_ok=True)
            output_path = str(plots_dir / f"edc_curve_room_{args.index}.png")
            predictor.plot_edc_curve(results['edc_predictions'], room_index=0, 
                                    sample_rate=args.sample_rate, save_path=output_path)
    
    elif args.indices:
        print(f"\n🔍 Predicting for rooms {args.indices}...")
        features = df_features.iloc[args.indices][numeric_cols].values
        results = predictor.predict_and_analyze(features)
        
        print(f"\n✓ Predictions successful!")
        print(f"  EDC shape: {results['edc_predictions'].shape}")
        for i, idx in enumerate(args.indices):
            print(f"\n  Room {idx}:")
            print(f"    T20: {results['t20_predictions'][i]:.4f} s")
            print(f"    C50: {results['c50_predictions'][i]:.4f} dB")
        
        # Plot EDC curves if requested
        if args.visualize:
            plots_dir = Path("edc_plots")
            plots_dir.mkdir(exist_ok=True)
            for i, idx in enumerate(args.indices):
                output_path = str(plots_dir / f"edc_curve_room_{idx}.png")
                predictor.plot_edc_curve(results['edc_predictions'], room_index=i,
                                        sample_rate=args.sample_rate, save_path=output_path)
    
    print("\n" + "="*60)
    print("✓ Inference completed successfully!")
    print("="*60 + "\n")


def evaluate_from_checkpoint(
    checkpoint_path: str,
    features_csv: str,
    edc_dir: str,
    numeric_cols: list
):
    """Evaluate model on test set."""
    from src.data.data_loader import load_edc_data
    
    # Load data
    edc_data = load_edc_data(edc_dir, max_samples=None)
    df_features = pd.read_csv(features_csv)
    
    # Filter to available samples
    available_indices = list(edc_data.keys())
    df_features_filtered = df_features.iloc[available_indices].reset_index(drop=True)
    
    # Initialize predictor
    predictor = EDCPredictor(checkpoint_path, features_csv)
    
    # Make predictions
    features = df_features_filtered[numeric_cols].values
    predictions = predictor.predict(features)
    
    # Prepare targets
    edc_targets = np.array([edc_data[idx] for idx in available_indices])
    
    # Evaluate
    print("\n📊 Evaluation Results:")
    print("-" * 60)
    
    edc_mae = np.mean(np.abs(predictions["edc"] - edc_targets))
    edc_rmse = np.sqrt(np.mean((predictions["edc"] - edc_targets) ** 2))
    edc_r2 = 1 - (np.sum((predictions["edc"] - edc_targets) ** 2) / 
                   np.sum((edc_targets - edc_targets.mean()) ** 2))
    
    print(f"EDC:")
    print(f"  MAE:  {edc_mae:.6f} (target: 0.020)")
    print(f"  RMSE: {edc_rmse:.6f} (target: 0.020)")
    print(f"  R²:   {edc_r2:.6f} (target: 0.980)")
    print("-" * 60)


if __name__ == "__main__":
    main()
