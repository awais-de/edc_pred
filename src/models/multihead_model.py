"""
Multi-Head CNN-LSTM model for simultaneous EDC, T20, and C50 prediction.
"""

import torch
import torch.nn as nn
from .base_model import BaseEDCModel


class CNNLSTMMultiHead(BaseEDCModel):
    """
    CNN-LSTM Multi-Head Model: Predicts EDC, T20, and C50 simultaneously.
    
    Architecture:
        Input → CNN pathway → LSTM pathway → Shared backbone
                ├→ EDC head (96k outputs)
                ├→ T20 head (1 output)
                └→ C50 head (1 output)
    
    Direct multi-task learning with explicit supervision on all 3 targets.
    """
    
    def __init__(
        self,
        input_dim: int,
        target_length: int,
        cnn_filters: list = None,
        cnn_kernel_sizes: list = None,
        lstm_hidden_dim: int = 128,
        fc_hidden_dim: int = 2048,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        edc_weight: float = 1.0,
        t20_weight: float = 1.0,
        c50_weight: float = 1.0
    ):
        """
        Initialize Multi-Head model.
        
        Args:
            input_dim: Number of input features
            target_length: Length of output EDC sequence
            cnn_filters: List of filter counts for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden_dim: Number of LSTM hidden units
            fc_hidden_dim: Number of fully connected hidden units
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            edc_weight: Weight for EDC loss component
            t20_weight: Weight for T20 loss component
            c50_weight: Weight for C50 loss component
        """
        super().__init__(input_dim, target_length, learning_rate)
        
        self.cnn_filters = cnn_filters or [32, 64]
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 3]
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout_rate
        self.edc_weight = edc_weight
        self.t20_weight = t20_weight
        self.c50_weight = c50_weight
        
        # Shared backbone: CNN pathway
        self.cnn_layers = nn.Sequential()
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(self.cnn_filters, self.cnn_kernel_sizes)):
            self.cnn_layers.add_module(
                f"conv1d_{i}",
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            self.cnn_layers.add_module(f"bn_{i}", nn.BatchNorm1d(out_channels))
            self.cnn_layers.add_module(f"relu_{i}", nn.ReLU())
            in_channels = out_channels
        
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Linear(in_channels, 256)
        
        # Shared backbone: LSTM pathway
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.lstm_fc = nn.Linear(lstm_hidden_dim, 256)
        
        # Shared representation
        self.dropout = nn.Dropout(dropout_rate)
        self.shared_fc = nn.Linear(512, fc_hidden_dim)  # 256 + 256
        
        # Task-specific heads
        # EDC head
        self.edc_fc1 = nn.Linear(fc_hidden_dim, fc_hidden_dim // 2)
        self.edc_fc2 = nn.Linear(fc_hidden_dim // 2, target_length)
        
        # T20 head
        self.t20_fc1 = nn.Linear(fc_hidden_dim, 256)
        self.t20_fc2 = nn.Linear(256, 1)
        
        # C50 head
        self.c50_fc1 = nn.Linear(fc_hidden_dim, 256)
        self.c50_fc2 = nn.Linear(256, 1)
        
        # Multi-task loss
        self.criterion = nn.MSELoss(reduction='none')  # We'll weight manually
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, input_dim)
            
        Returns:
            Dictionary with keys 'edc', 't20', 'c50'
        """
        # CNN pathway
        cnn_out = self.cnn_layers(x)  # shape: (batch_size, channels, input_dim)
        cnn_out = self.cnn_pool(cnn_out)  # shape: (batch_size, channels, 1)
        cnn_out = cnn_out.view(cnn_out.shape[0], -1)  # shape: (batch_size, channels)
        cnn_out = torch.relu(self.cnn_fc(cnn_out))  # shape: (batch_size, 256)
        
        # LSTM pathway
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out = h_n[-1]  # shape: (batch_size, lstm_hidden_dim)
        lstm_out = torch.relu(self.lstm_fc(lstm_out))  # shape: (batch_size, 256)
        
        # Merge pathways
        merged = torch.cat([cnn_out, lstm_out], dim=1)  # shape: (batch_size, 512)
        
        # Shared representation
        shared = torch.relu(self.shared_fc(merged))  # shape: (batch_size, fc_hidden_dim)
        shared = self.dropout(shared)
        
        # EDC head
        edc = torch.relu(self.edc_fc1(shared))
        edc = self.dropout(edc)
        edc_out = self.edc_fc2(edc)  # shape: (batch_size, target_length)
        
        # T20 head
        t20 = torch.relu(self.t20_fc1(shared))
        t20 = self.dropout(t20)
        t20_out = self.t20_fc2(t20)  # shape: (batch_size, 1)
        
        # C50 head
        c50 = torch.relu(self.c50_fc1(shared))
        c50 = self.dropout(c50)
        c50_out = self.c50_fc2(c50)  # shape: (batch_size, 1)
        
        return {
            'edc': edc_out,
            't20': t20_out.squeeze(-1),  # shape: (batch_size,)
            'c50': c50_out.squeeze(-1)   # shape: (batch_size,)
        }
    
    def training_step(self, batch, batch_idx):
        """Training step for multi-output."""
        x, y_edc, y_t20, y_c50 = batch
        
        # Forward pass
        outputs = self(x)
        
        # Compute losses
        loss_edc = self.criterion(outputs['edc'], y_edc).mean()
        loss_t20 = self.criterion(outputs['t20'], y_t20).mean()
        loss_c50 = self.criterion(outputs['c50'], y_c50).mean()
        
        # Weighted combination
        loss = (self.edc_weight * loss_edc + 
                self.t20_weight * loss_t20 + 
                self.c50_weight * loss_c50)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss_edc', loss_edc, prog_bar=False)
        self.log('train_loss_t20', loss_t20, prog_bar=False)
        self.log('train_loss_c50', loss_c50, prog_bar=False)
        
        # MAE for EDC
        mae_edc = torch.abs(outputs['edc'] - y_edc).mean()
        mae_t20 = torch.abs(outputs['t20'] - y_t20).mean()
        mae_c50 = torch.abs(outputs['c50'] - y_c50).mean()
        
        self.log('train_mae_edc', mae_edc, prog_bar=False)
        self.log('train_mae_t20', mae_t20, prog_bar=False)
        self.log('train_mae_c50', mae_c50, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for multi-output."""
        x, y_edc, y_t20, y_c50 = batch
        
        # Forward pass
        outputs = self(x)
        
        # Compute losses
        loss_edc = self.criterion(outputs['edc'], y_edc).mean()
        loss_t20 = self.criterion(outputs['t20'], y_t20).mean()
        loss_c50 = self.criterion(outputs['c50'], y_c50).mean()
        
        # Weighted combination
        loss = (self.edc_weight * loss_edc + 
                self.t20_weight * loss_t20 + 
                self.c50_weight * loss_c50)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_loss_edc', loss_edc, prog_bar=False)
        self.log('val_loss_t20', loss_t20, prog_bar=False)
        self.log('val_loss_c50', loss_c50, prog_bar=False)
        
        # MAE for all outputs
        mae_edc = torch.abs(outputs['edc'] - y_edc).mean()
        mae_t20 = torch.abs(outputs['t20'] - y_t20).mean()
        mae_c50 = torch.abs(outputs['c50'] - y_c50).mean()
        
        self.log('val_mae_edc', mae_edc, prog_bar=True)
        self.log('val_mae_t20', mae_t20, prog_bar=True)
        self.log('val_mae_c50', mae_c50, prog_bar=True)
        
        return loss
