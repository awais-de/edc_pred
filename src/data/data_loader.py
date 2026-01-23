"""
Data loading and preprocessing utilities.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, Optional


class EDCDataset(Dataset):
    """PyTorch Dataset for EDC data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input features (room features)
            y: Target EDCs
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        assert len(self.X) == len(self.y), "Mismatched data lengths"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EDCMultiOutputDataset(Dataset):
    """PyTorch Dataset for multi-output EDC data (EDC + T20 + C50)."""
    
    def __init__(self, X: np.ndarray, y_edc: np.ndarray, y_t20: np.ndarray, y_c50: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input features (room features)
            y_edc: Target EDCs
            y_t20: Target T20 values
            y_c50: Target C50 values
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_edc = torch.tensor(y_edc, dtype=torch.float32)
        self.y_t20 = torch.tensor(y_t20, dtype=torch.float32)
        self.y_c50 = torch.tensor(y_c50, dtype=torch.float32)
        
        assert len(self.X) == len(self.y_edc) == len(self.y_t20) == len(self.y_c50), "Mismatched data lengths"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_edc[idx], self.y_t20[idx], self.y_c50[idx]


def compute_t20_c50_from_edc(edc: np.ndarray, sample_rate: int = 48000) -> Tuple[float, float]:
    """
    Compute T20 and C50 from EDC curve.
    
    Args:
        edc: Energy Decay Curve (1D array)
        sample_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (t20, c50) values
    """
    # Normalize EDC to start at 0 dB
    if edc.max() <= 0:
        edc_db = edc.copy()
    else:
        edc_db = 10 * np.log10(np.maximum(edc, 1e-10) / edc.max())
    
    time_s = np.arange(len(edc)) / sample_rate
    
    # T20: Time to decay from -5 to -25 dB (extrapolated to 60 dB)
    idx_5db = np.where(edc_db <= -5)[0]
    idx_25db = np.where(edc_db <= -25)[0]
    
    if len(idx_5db) > 0 and len(idx_25db) > 0:
        time_5db = time_s[idx_5db[0]]
        time_25db = time_s[idx_25db[0]]
        t20 = 3 * (time_25db - time_5db)
    else:
        t20 = 0.0  # Default fallback
    
    # C50: Clarity index (energy in first 50ms / total energy after 50ms)
    idx_50ms = np.argmin(np.abs(time_s - 0.05))
    early_energy = np.sum(edc[:idx_50ms])
    late_energy = np.sum(edc[idx_50ms:])
    
    if late_energy > 0 and early_energy > 0:
        c50 = 10 * np.log10(early_energy / late_energy)
    else:
        c50 = 0.0  # Default fallback
    
    return t20, c50


def extract_rir_case(fname: str) -> Tuple[int, int]:
    """
    Extract RIR number and case number from filename.
    
    Args:
        fname: Filename like "RIR_001_case0_edc.npy"
        
    Returns:
        Tuple of (rir_num, case_num) for sorting
    """
    rir_match = re.search(r'RIR_(\d+)', fname)
    case_match = re.search(r'case(\d+)', fname)
    rir_num = int(rir_match.group(1)) if rir_match else float('inf')
    case_num = int(case_match.group(1)) if case_match else float('inf')
    return (rir_num, case_num)


def load_edc_data(
    edc_folder_path: str,
    target_length: int = 96000,
    max_files: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Load EDC data from folder.
    
    Args:
        edc_folder_path: Path to folder containing .npy EDC files
        target_length: Target sequence length for EDCs
        max_files: Maximum number of files to load
        verbose: Print loading progress
        
    Returns:
        Array of shape (num_samples, target_length)
    """
    edc_files = sorted(
        [f for f in os.listdir(edc_folder_path) if f.endswith('.npy')],
        key=extract_rir_case
    )
    
    if max_files:
        edc_files = edc_files[:max_files]
    
    all_edcs = []
    for i, fname in enumerate(edc_files):
        try:
            edc = np.load(os.path.join(edc_folder_path, fname))
            edc = edc.flatten()[:target_length]
            edc = np.pad(edc, (0, max(0, target_length - len(edc))), mode='constant')
            all_edcs.append(edc)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(edc_files)} files")
        except Exception as e:
            if verbose:
                print(f"Failed to load {fname}: {e}")
    
    if verbose:
        print(f"Successfully loaded {len(all_edcs)} EDC files")
    
    return np.stack(all_edcs)


def load_room_features(
    csv_path: str,
    max_samples: Optional[int] = None,
    drop_id: bool = True
) -> np.ndarray:
    """
    Load room features from CSV.
    
    Args:
        csv_path: Path to room features CSV
        max_samples: Maximum number of samples to load
        drop_id: Whether to drop ID column
        
    Returns:
        Array of shape (num_samples, num_features)
    """
    df = pd.read_csv(csv_path)
    
    if drop_id and 'ID' in df.columns:
        df = df.drop(columns=['ID'], errors='ignore')
    
    features = df.values
    if max_samples:
        features = features[:max_samples]
    
    return features


def scale_data(
    X: np.ndarray,
    y: np.ndarray,
    scaler_type: str = "minmax",
    scaler_X = None,
    scaler_y = None
) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """
    Scale input and output data.
    
    Args:
        X: Input features
        y: Target values
        scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        scaler_X: Existing X scaler (for inference)
        scaler_y: Existing y scaler (for inference)
        
    Returns:
        Tuple of (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    if scaler_type == "minmax":
        ScalerClass = MinMaxScaler
    elif scaler_type == "standard":
        ScalerClass = StandardScaler
    elif scaler_type == "robust":
        ScalerClass = RobustScaler
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    if scaler_X is None:
        scaler_X = ScalerClass()
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = scaler_X.transform(X)
    
    if scaler_y is None:
        scaler_y = ScalerClass()
        y_scaled = scaler_y.fit_transform(y)
    else:
        y_scaled = scaler_y.transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_state: int = 42,
    input_reshape: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        X: Input features
        y: Target values
        batch_size: Batch size
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (remainder is test)
        random_state: Random seed
        input_reshape: Whether to reshape X to (batch, 1, features)
        num_workers: Dataloader workers for background loading
        pin_memory: Pin CPU memory for faster H2D transfer
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Reshape input if needed
    if input_reshape and len(X.shape) == 2:
        X = X.reshape((-1, 1, X.shape[1]))
    
    # Train/val/test split
    test_ratio = 1.0 - train_ratio - val_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state
    )
    
    # Create datasets
    train_dataset = EDCDataset(X_train, y_train)
    val_dataset = EDCDataset(X_val, y_val)
    test_dataset = EDCDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    
    return train_loader, val_loader, test_loader


def create_multioutput_dataloaders(
    X: np.ndarray,
    y_edc: np.ndarray,
    batch_size: int = 8,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_state: int = 42,
    input_reshape: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    sample_rate: int = 48000,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for multi-output (EDC + T20 + C50).
    
    Args:
        X: Input features
        y_edc: Target EDC curves
        batch_size: Batch size
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (remainder is test)
        random_state: Random seed
        input_reshape: Whether to reshape X to (batch, 1, features)
        num_workers: Dataloader workers for background loading
        pin_memory: Pin CPU memory for faster H2D transfer
        persistent_workers: Keep workers alive between epochs
        sample_rate: Sampling rate for acoustic parameter calculation
        verbose: Print progress
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if verbose:
        print("Computing T20 and C50 values from EDC curves...")
    
    # Compute T20 and C50 for all samples
    y_t20 = []
    y_c50 = []
    
    for i, edc in enumerate(y_edc):
        t20, c50 = compute_t20_c50_from_edc(edc, sample_rate)
        y_t20.append(t20)
        y_c50.append(c50)
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(y_edc)} EDC curves")
    
    y_t20 = np.array(y_t20, dtype=np.float32)
    y_c50 = np.array(y_c50, dtype=np.float32)
    
    if verbose:
        print(f"  T20 range: [{y_t20.min():.4f}, {y_t20.max():.4f}], mean: {y_t20.mean():.4f}")
        print(f"  C50 range: [{y_c50.min():.4f}, {y_c50.max():.4f}], mean: {y_c50.mean():.4f}")
    
    # Reshape input if needed
    if input_reshape and len(X.shape) == 2:
        X = X.reshape((-1, 1, X.shape[1]))
    
    # Train/val/test split
    test_ratio = 1.0 - train_ratio - val_ratio
    X_temp, X_test, y_edc_temp, y_edc_test, y_t20_temp, y_t20_test, y_c50_temp, y_c50_test = train_test_split(
        X, y_edc, y_t20, y_c50, test_size=test_ratio, random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_edc_train, y_edc_val, y_t20_train, y_t20_val, y_c50_train, y_c50_val = train_test_split(
        X_temp, y_edc_temp, y_t20_temp, y_c50_temp, test_size=val_ratio_adjusted, random_state=random_state
    )
    
    # Create datasets
    train_dataset = EDCMultiOutputDataset(X_train, y_edc_train, y_t20_train, y_c50_train)
    val_dataset = EDCMultiOutputDataset(X_val, y_edc_val, y_t20_val, y_c50_val)
    test_dataset = EDCMultiOutputDataset(X_test, y_edc_test, y_t20_test, y_c50_test)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    
    return train_loader, val_loader, test_loader

