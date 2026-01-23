"""
Training script for multi-head EDC prediction model (EDC + T20 + C50).

This model directly predicts T20 and C50 alongside the EDC curve for better supervision.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.models import get_model
from src.data.data_loader import (
    load_edc_data, load_room_features, scale_data, create_multioutput_dataloaders
)
from src.evaluation.metrics import evaluate_multioutput_model, print_metrics


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Multi-Head EDC prediction model")
    
    parser.add_argument(
        "--edc-path", type=str, default="data/raw/EDC",
        help="Path to EDC dataset"
    )
    parser.add_argument(
        "--features-path", type=str, default="data/raw/roomFeaturesDataset.csv",
        help="Path to room features CSV"
    )
    parser.add_argument(
        "--max-samples", type=int, default=600,
        help="Maximum number of samples to load"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="Optional batch size for val/test"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--pin-memory", action="store_true",
        help="Pin CPU memory for faster GPU transfer"
    )
    parser.add_argument(
        "--persistent-workers", action="store_true",
        help="Keep workers alive between epochs"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Maximum epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--scaler-type", type=str, default="standard",
        choices=["minmax", "standard", "robust"],
        help="Scaler type"
    )
    parser.add_argument(
        "--precision", type=int, default=32,
        choices=[16, 32],
        help="Training precision"
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true",
        help="Disable mixed precision (force 32-bit)"
    )
    parser.add_argument(
        "--gradient-clip-val", type=float, default=1.0,
        help="Gradient clipping value (None to disable)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments",
        help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6,
        help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=50,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--disable-early-stop", action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--edc-weight", type=float, default=1.0,
        help="Weight for EDC loss component"
    )
    parser.add_argument(
        "--t20-weight", type=float, default=100.0,
        help="Weight for T20 loss component (higher = more focus on T20)"
    )
    parser.add_argument(
        "--c50-weight", type=float, default=50.0,
        help="Weight for C50 loss component (higher = more focus on C50)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"multihead_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MULTI-HEAD EDC PREDICTION MODEL")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # ===== DATA LOADING =====
    print("Step 1: Loading data...")
    edc_data = load_edc_data(args.edc_path, max_files=args.max_samples)
    room_features = load_room_features(args.features_path, max_samples=args.max_samples)
    
    # Verify alignment
    num_samples = min(len(edc_data), len(room_features))
    edc_data = edc_data[:num_samples]
    room_features = room_features[:num_samples]
    
    print(f"✓ Loaded {num_samples} EDC samples in {time.time():.2f}s")
    print(f"  EDC shape: {edc_data.shape}")
    print(f"  Features shape: {room_features.shape}\n")
    
    # ===== SCALING =====
    print("Step 2: Scaling data...")
    room_features_scaled, edc_data_scaled, scaler_X, scaler_y = scale_data(
        room_features, edc_data, scaler_type=args.scaler_type
    )
    print(f"✓ Data scaled using {args.scaler_type} scaler\n")
    
    # ===== CREATE DATALOADERS (with T20/C50 computation) =====
    print("Step 3: Creating multi-output dataloaders...")
    train_loader, val_loader, test_loader = create_multioutput_dataloaders(
        room_features_scaled,
        edc_data,  # Use unscaled EDC for acoustic parameter calculation
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_reshape=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        verbose=True
    )
    
    print(f"✓ Created dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # ===== MODEL INITIALIZATION =====
    print("Step 4: Initializing multi-head model...")
    model_kwargs = {
        "input_dim": room_features.shape[1],
        "target_length": edc_data.shape[1],
        "learning_rate": args.learning_rate,
        "edc_weight": args.edc_weight,
        "t20_weight": args.t20_weight,
        "c50_weight": args.c50_weight,
    }
    
    model = get_model("multihead", **model_kwargs)
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Initialized multihead model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Loss weights: EDC={args.edc_weight}, T20={args.t20_weight}, C50={args.c50_weight}\n")
    
    # ===== TRAINING =====
    print("Step 5: Training model...")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    if args.gradient_clip_val is not None:
        print(f"  Gradient clipping: {args.gradient_clip_val}")
    if args.no_mixed_precision:
        print(f"  Precision: 32-bit (mixed precision disabled)\n")
    else:
        print(f"  Precision: {args.precision}-bit\n")
    
    # Setup callbacks
    callbacks = []
    
    if not args.disable_early_stop:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            mode="min",
            verbose=False
        )
        callbacks.append(early_stop)
    
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        verbose=False
    )
    callbacks.append(checkpoint)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs"
    )
    
    # Create trainer
    precision_to_use = 32 if args.no_mixed_precision else args.precision
    gradient_clip = args.gradient_clip_val if args.gradient_clip_val is not None else None
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device if args.device != "auto" else "auto",
        devices="auto",
        precision=precision_to_use,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=gradient_clip
    )
    
    train_start = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_duration = time.time() - train_start
    
    print(f"✓ Training completed in {train_duration/60:.2f} minutes\n")
    
    # ===== EVALUATION =====
    print("Step 6: Evaluating on test set...")
    
    model.eval()
    edc_preds = []
    edc_targets = []
    t20_preds = []
    t20_targets = []
    c50_preds = []
    c50_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            X, y_edc, y_t20, y_c50 = batch
            outputs = model(X)
            
            edc_preds.append(outputs['edc'].cpu().numpy())
            t20_preds.append(outputs['t20'].cpu().numpy())
            c50_preds.append(outputs['c50'].cpu().numpy())
            
            edc_targets.append(y_edc.numpy())
            t20_targets.append(y_t20.numpy())
            c50_targets.append(y_c50.numpy())
    
    edc_preds = np.vstack(edc_preds)
    edc_targets = np.vstack(edc_targets)
    t20_preds = np.concatenate(t20_preds)
    t20_targets = np.concatenate(t20_targets)
    c50_preds = np.concatenate(c50_preds)
    c50_targets = np.concatenate(c50_targets)
    
    # Inverse scale EDC only (T20/C50 already in original scale)
    edc_preds_rescaled = scaler_y.inverse_transform(edc_preds)
    edc_targets_rescaled = scaler_y.inverse_transform(edc_targets)
    
    # Compute metrics
    metrics = evaluate_multioutput_model(
        edc_targets_rescaled, edc_preds_rescaled,
        t20_targets, t20_preds,
        c50_targets, c50_preds
    )
    
    print_metrics(metrics)
    
    # ===== SAVE RESULTS =====
    print("Step 7: Saving results...")
    
    # Save predictions
    np.save(os.path.join(output_dir, "edc_predictions.npy"), edc_preds_rescaled)
    np.save(os.path.join(output_dir, "edc_targets.npy"), edc_targets_rescaled)
    np.save(os.path.join(output_dir, "t20_predictions.npy"), t20_preds)
    np.save(os.path.join(output_dir, "t20_targets.npy"), t20_targets)
    np.save(os.path.join(output_dir, "c50_predictions.npy"), c50_preds)
    np.save(os.path.join(output_dir, "c50_targets.npy"), c50_targets)
    
    # Save scalers
    import joblib
    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))
    
    # Convert metrics to native Python types
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    # Save metadata
    metadata = {
        "model_name": "multihead",
        "timestamp": timestamp,
        "model_parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params)
        },
        "training_config": {
            "max_epochs": int(args.max_epochs),
            "actual_epochs": int(trainer.current_epoch + 1),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "training_duration_seconds": float(train_duration),
            "training_duration_minutes": float(train_duration / 60)
        },
        "loss_config": {
            "edc_weight": float(args.edc_weight),
            "t20_weight": float(args.t20_weight),
            "c50_weight": float(args.c50_weight),
            "gradient_clip_val": args.gradient_clip_val
        },
        "precision_config": {
            "precision": int(precision_to_use),
            "mixed_precision_enabled": not args.no_mixed_precision
        },
        "data_config": {
            "num_samples": int(num_samples),
            "input_dim": int(room_features.shape[1]),
            "output_length": int(edc_data.shape[1]),
            "scaler_type": args.scaler_type
        },
        "metrics": metrics_serializable,
        "best_model_path": str(checkpoint.best_model_path)
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Results saved to {output_dir}\n")
    
    print(f"{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")
    
    return output_dir


if __name__ == "__main__":
    main()
