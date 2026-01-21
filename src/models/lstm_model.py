"""
LSTM model for EDC prediction.
"""

import torch
import torch.nn as nn
from .base_model import BaseEDCModel


class LSTMModel(BaseEDCModel):
    """
    LSTM-based model for EDC prediction from room features.
    
    Architecture:
        Input → LSTM → FC1 → Dropout → FC2 → Output
    """
    
    def __init__(
        self, 
        input_dim: int, 
        target_length: int,
        lstm_hidden_dim: int = 128,
        fc_hidden_dim: int = 2048,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        loss_type: str = "mse"
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            target_length: Length of output EDC sequence
            lstm_hidden_dim: Number of LSTM hidden units
            fc_hidden_dim: Number of fully connected hidden units
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function ('mse', 'edc_rir')
        """
        super().__init__(input_dim, target_length, learning_rate)
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        
        # Architecture
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, target_length)
        
        # Loss function
        self._build_criterion()
    
    def _build_criterion(self):
        """Build loss function."""
        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "edc_rir":
            self.criterion = EDCRIRLoss(alpha=1.0, beta=0.5)
        elif self.loss_type == "weighted_edc":
            self.criterion = WeightedEDCLoss(
                sampling_rate=48000,
                edt_weight=2.0,
                t20_weight=3.0,
                c50_weight=3.0,
                base_weight=1.0
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, target_length)
        """
        # LSTM forward
        _, (h_n, _) = self.lstm(x)  # h_n shape: (1, batch_size, lstm_hidden_dim)
        
        # Take last hidden state
        x = h_n[-1]  # shape: (batch_size, lstm_hidden_dim)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))  # shape: (batch_size, fc_hidden_dim)
        x = self.dropout(x)
        x = self.fc2(x)  # shape: (batch_size, target_length)
        
        return x


class EDCRIRLoss(nn.Module):
    """
    Combined EDC and RIR loss.
    
    Loss = alpha * MSE(EDC) + beta * MSE(RIR_derivative)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Initialize loss.
        
        Args:
            alpha: Weight for EDC loss
            beta: Weight for RIR loss
        """
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, return_components: bool = False):
        """
        Compute combined loss.
        
        Args:
            y_pred: Predicted EDC
            y_true: Ground truth EDC
            return_components: If True, return individual components
            
        Returns:
            Loss (or tuple of losses if return_components=True)
        """
        # EDC loss
        edc_loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction='mean')
        
        # RIR loss (derivative of EDC)
        rir_pred = y_pred[:, 1:] - y_pred[:, :-1]
        rir_true = y_true[:, 1:] - y_true[:, :-1]
        rir_loss = torch.nn.functional.mse_loss(rir_pred, rir_true, reduction='mean')
        
        # Combined loss
        total = self.alpha * edc_loss + self.beta * rir_loss
        
        if return_components:
            return total, edc_loss, rir_loss
        return total


class WeightedEDCLoss(nn.Module):
    """
    Region-weighted EDC loss that emphasizes T20 and C50 critical zones.
    
    Weights different EDC regions based on their importance for acoustic parameters:
    - EDT region (0 to -10dB): Weight 1.5
    - T20 region (-5dB to -25dB): Weight 1.5 (critical)
    - C50 early region (first 50ms): Weight 1.5 (critical)
    - Late decay: Weight 1.0 (baseline)
    """
    
    def __init__(
        self, 
        sampling_rate: int = 48000,
        edt_weight: float = 2.0,
        t20_weight: float = 2.0,
        c50_weight: float = 2.0,
        base_weight: float = 1.0
    ):
        """
        Initialize weighted loss.
        
        Args:
            sampling_rate: Sampling rate of EDC (default 48kHz)
            edt_weight: Weight for EDT region (0 to -10dB)
            t20_weight: Weight for T20 region (-5dB to -25dB)
            c50_weight: Weight for C50 early region (first 50ms)
            base_weight: Weight for remaining regions
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.edt_weight = edt_weight
        self.t20_weight = t20_weight
        self.c50_weight = c50_weight
        self.base_weight = base_weight
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            y_pred: Predicted EDC (batch_size, seq_len) in linear scale
            y_true: Ground truth EDC (batch_size, seq_len) in linear scale
            
        Returns:
            Weighted MSE loss
        """
        batch_size, seq_len = y_pred.shape
        
        # Convert to dB scale for zone identification (add small epsilon to avoid log(0))
        eps = 1e-10
        y_true_db = 10 * torch.log10(y_true + eps)
        
        # Normalize to 0dB at peak
        y_true_db = y_true_db - y_true_db.max(dim=1, keepdim=True)[0]
        
        # Initialize weight mask with base weight
        weight_mask = torch.ones_like(y_pred) * self.base_weight
        
        # C50 region: First 50ms (critical for clarity)
        c50_samples = int(0.050 * self.sampling_rate)  # 50ms
        if c50_samples < seq_len:
            weight_mask[:, :c50_samples] = self.c50_weight
        
        # EDT region: 0 to -10dB
        edt_mask = (y_true_db >= -10.0)
        weight_mask[edt_mask] = torch.maximum(weight_mask[edt_mask], 
                                               torch.tensor(self.edt_weight).to(weight_mask.device))
        
        # T20 region: -5dB to -25dB (most critical for reverberation time)
        t20_mask = (y_true_db >= -25.0) & (y_true_db <= -5.0)
        weight_mask[t20_mask] = torch.maximum(weight_mask[t20_mask], 
                                               torch.tensor(self.t20_weight).to(weight_mask.device))
        
        # Compute weighted MSE
        squared_error = (y_pred - y_true) ** 2
        weighted_error = squared_error * weight_mask
        
        # Mean over all elements
        loss = weighted_error.mean()
        
        return loss
