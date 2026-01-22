"""
Example training script for EDC prediction models.

This script demonstrates how to:
1. Load and preprocess data
2. Initialize a model (LSTM or hybrid)
3. Train using PyTorch Lightning
4. Evaluate and compare architectures
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

from src.models import get_model, list_available_models
from src.data.data_loader import (
    load_edc_data, load_room_features, scale_data, create_dataloaders
)
from src.evaluation.metrics import evaluate_model, print_metrics


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train EDC prediction model")
    
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=list_available_models(),
        help="Model architecture to train"
    )
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
        help="Optional batch size for val/test to reduce peak memory (defaults to train batch size)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader workers (0 = main process)"
    )
    parser.add_argument(
        "--pin-memory", action="store_true",
        help="Pin CPU memory for faster host-to-GPU transfer"
    )
    parser.add_argument(
        "--persistent-workers", action="store_true",
        help="Keep DataLoader workers alive across epochs (requires num-workers > 0)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--scaler-type", type=str, default="minmax",
        choices=["minmax", "standard", "robust"],
        help="Scaler for inputs/targets"
    )
    parser.add_argument(
        "--precision", type=int, default=32,
        choices=[16, 32],
        help="Training precision (16 = mixed precision, 32 = full precision)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--loss-type", type=str, default="mse",
        choices=["mse", "edc_rir", "weighted_edc", "auxiliary"],
        help="Loss function type: mse (MSE), edc_rir (EDC+RIR weighted loss), weighted_edc (region-weighted EDC loss), or auxiliary (MSE + T20/C50 supervision)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Proportion of data for validation"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=15,
        help="Early stopping patience (epochs without val_loss improvement)"
    )
    parser.add_argument(
        "--disable-early-stop", action="store_true",
        help="Disable early stopping to force training for max_epochs"
    )
    parser.add_argument(
        "--gradient-clip-val", type=float, default=None,
        help="Gradient clipping threshold (None to disable); recommended 1.0-10.0 for stability"
    )
    parser.add_argument(
        "--aux-weight", type=float, default=0.1,
        help="Weight for auxiliary T20/C50 losses when using --loss-type auxiliary (default 0.1)"
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true",
        help="Disable mixed precision (force 32-bit precision) for debugging numerical issues"
    )
    parser.add_argument(
        "--edt-weight", type=float, default=1.5,
        help="Weight for EDT region (0 to -10dB) in weighted_edc loss (default 1.5)"
    )
    parser.add_argument(
        "--t20-weight", type=float, default=1.5,
        help="Weight for T20 region (-5dB to -25dB) in weighted_edc loss (default 1.5)"
    )
    parser.add_argument(
        "--c50-weight", type=float, default=1.5,
        help="Weight for C50 early region (first 50ms) in weighted_edc loss (default 1.5)"
    )
    
    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EDC PREDICTION - {args.model.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}\n")
    
    # ===== DATA LOADING =====
    print("Step 1: Loading data...")
    start_time = time.time()
    
    edc_data = load_edc_data(
        args.edc_path,
        target_length=96000,
        max_files=args.max_samples,
        verbose=True
    )
    room_features = load_room_features(
        args.features_path,
        max_samples=args.max_samples
    )

    # Align counts in case features < EDCs (or vice versa)
    n_samples = min(len(edc_data), len(room_features))
    edc_data = edc_data[:n_samples]
    room_features = room_features[:n_samples]

    print(f"✓ Loaded {len(edc_data)} EDC samples in {time.time()-start_time:.2f}s")
    print(f"  EDC shape: {edc_data.shape}")
    print(f"  Features shape: {room_features.shape}\n")
    
    # ===== DATA SCALING =====
    print("Step 2: Scaling data...")
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(
        room_features, edc_data, scaler_type=args.scaler_type
    )
    print(f"✓ Data scaled using {args.scaler_type} scaler\n")
    
    # ===== DATALOADERS =====
    print("Step 3: Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_scaled, y_scaled,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_reshape=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # Optionally use a smaller batch size for evaluation to reduce peak memory
    eval_batch_size = args.eval_batch_size or args.batch_size
    if eval_batch_size != args.batch_size:
        val_loader.batch_size = eval_batch_size
        test_loader.batch_size = eval_batch_size
    
    print(f"✓ Created dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}\n")
    
    # ===== MODEL INITIALIZATION =====
    print("Step 4: Initializing model...")
    
    model_kwargs = {
        "input_dim": room_features.shape[1],
        "target_length": edc_data.shape[1],
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type
    }
    
    # Add architecture-specific parameters
    if "transformer" in args.model:
        # Transformer has different parameters
        model_kwargs.update({
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "ff_dim": 1024,
            "dropout_rate": 0.1
        })
    elif "hybrid_v3" in args.model:
        # hybrid_v3 doesn't take cnn_filters/cnn_kernel_sizes
        model_kwargs.update({
            "lstm_hidden_dim": 128,
            "fc_hidden_dim": 2048,
            "dropout_rate": 0.3
        })
    elif "hybrid" in args.model:
        model_kwargs.update({
            "cnn_filters": [32, 64],
            "cnn_kernel_sizes": [3, 3],
            "lstm_hidden_dim": 128,
            "fc_hidden_dim": 2048,
            "dropout_rate": 0.3
        })
    else:
        # LSTM model
        model_kwargs.update({
            "lstm_hidden_dim": 128,
            "fc_hidden_dim": 2048,
            "dropout_rate": 0.3
        })
    
    model = get_model(args.model, **model_kwargs)
    
    # Patch aux_weight if using auxiliary loss
    if args.loss_type == "auxiliary" and hasattr(model.criterion, 'aux_weight'):
        model.criterion.aux_weight = args.aux_weight
        print(f"  Auxiliary loss weight set to: {args.aux_weight}")
        # Patch weighted_edc weights if specified via CLI
    if args.loss_type == "weighted_edc" and hasattr(model.criterion, 'edt_weight'):
        model.criterion.edt_weight = args.edt_weight
        model.criterion.t20_weight = args.t20_weight
        model.criterion.c50_weight = args.c50_weight
        print(f"  Weighted EDC loss weights set to: EDT={args.edt_weight}, T20={args.t20_weight}, C50={args.c50_weight}")
        total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Initialized {args.model} model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
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
            verbose=True
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
    preds = []
    targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            preds.append(output.cpu().numpy())
            targets.append(y.numpy())
    
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    
    # Inverse scale
    preds_rescaled = scaler_y.inverse_transform(preds)
    targets_rescaled = scaler_y.inverse_transform(targets)
    
    # Compute metrics
    metrics = evaluate_model(targets_rescaled, preds_rescaled, compute_acoustic=True)
    
    print_metrics(metrics)
    
    # ===== SAVE RESULTS =====
    print("Step 7: Saving results...")
    
    # Save predictions
    np.save(os.path.join(output_dir, "predictions.npy"), preds_rescaled)
    np.save(os.path.join(output_dir, "targets.npy"), targets_rescaled)
    
    # Save scalers
    import joblib
    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))
    
    # Convert metrics to native Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    # Save metadata
    metadata = {
        "model_name": args.model,
        "timestamp": timestamp,
        "model_parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params)
        },
        "training_config": {
            "max_epochs": int(args.max_epochs),
            "actual_epochs": int(trainer.current_epoch + 1),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size) if args.eval_batch_size else int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "training_duration_seconds": float(train_duration),
            "training_duration_minutes": float(train_duration / 60)
        },
        "loss_config": {
            "loss_type": args.loss_type,
            "gradient_clip_val": args.gradient_clip_val,
            "aux_weight": args.aux_weight if args.loss_type == "auxiliary" else None,
            "edt_weight": args.edt_weight if args.loss_type == "weighted_edc" else None,
            "t20_weight": args.t20_weight if args.loss_type == "weighted_edc" else None,
            "c50_weight": args.c50_weight if args.loss_type == "weighted_edc" else None
        },
        "precision_config": {
            "precision": int(precision_to_use),
            "mixed_precision_enabled": not args.no_mixed_precision
        },
        "data_loader_config": {
            "num_workers": int(args.num_workers),
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "scaler_type": args.scaler_type
        },
        "data_config": {
            "num_samples": int(args.max_samples),
            "input_dim": int(room_features.shape[1]),
            "output_length": int(edc_data.shape[1]),
            "train_size": int(len(train_loader.dataset)),
            "val_size": int(len(val_loader.dataset)),
            "test_size": int(len(test_loader.dataset)),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio)
        },
        "early_stopping_config": {
            "enabled": not args.disable_early_stop,
            "patience": int(args.early_stop_patience) if not args.disable_early_stop else None,
            "stopped_at_epoch": int(trainer.current_epoch + 1) if (trainer.current_epoch + 1) < args.max_epochs else args.max_epochs
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
