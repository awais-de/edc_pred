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
    input_reshape: bool = True
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
