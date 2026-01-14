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
