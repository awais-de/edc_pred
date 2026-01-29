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
        
        # Load scaler from checkpoint directory
        checkpoint_dir = Path(checkpoint_path).parent
        scaler_path = checkpoint_dir / "scaler.pkl"
        
        if not scaler_path.exists():
            print(f"⚠️  Scaler not found at {scaler_path}")
            print("   Will attempt to create scaler from features CSV")
            self.scaler = self._load_or_create_scaler(features_csv)
        else:
            import joblib
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded scaler from {scaler_path}")
        
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
        if "hyper_parameters" in checkpoint:
            config = checkpoint["hyper_parameters"]
        else:
            # Fallback: use metadata from experiment directory
            experiment_dir = Path(self.checkpoint_path).parent.parent
            metadata_path = experiment_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                config = {
                    "input_dim": metadata.get("data_config", {}).get("input_dim", 16),
                    "target_length": metadata.get("data_config", {}).get("output_length", 96000),
                }
            else:
                # Default config
                config = {"input_dim": 16, "target_length": 96000}
        
        # Instantiate model
        model = get_model(
            model_name="multihead",
            input_dim=config.get("input_dim", 16),
            target_length=config.get("target_length", 96000),
            learning_rate=config.get("learning_rate", 0.001),
            **{k: v for k, v in config.items() 
               if k not in ["input_dim", "target_length", "learning_rate"]}
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
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(room_features_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            if isinstance(outputs, (list, tuple)):
                edc_pred, t20_pred, c50_pred = outputs
            else:
                # Handle case where model returns dict
                edc_pred = outputs.get("edc", outputs[0])
                t20_pred = outputs.get("t20", outputs[1])
                c50_pred = outputs.get("c50", outputs[2])
        
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
