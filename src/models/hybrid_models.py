"""
CNN-LSTM Hybrid models for EDC prediction.
"""

import torch
import torch.nn as nn
from .lstm_model import EDCRIRLoss, WeightedEDCLoss
from .base_model import BaseEDCModel


class CNNLSTMHybridV1(BaseEDCModel):
    """
    CNN-LSTM Hybrid Model V1: Sequential CNN→LSTM.
    
    Architecture:
        Input → Conv1D layers → LSTM → FC layers → Output
    
    The CNN extracts feature patterns from the input features,
    then LSTM models temporal sequence generation.
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
        loss_type: str = "mse"
    ):
        """
        Initialize CNN-LSTM Hybrid V1.
        
        Args:
            input_dim: Number of input features
            target_length: Length of output EDC sequence
            cnn_filters: List of filter counts for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden_dim: Number of LSTM hidden units
            fc_hidden_dim: Number of fully connected hidden units
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function ('mse', 'edc_rir')
        """
        super().__init__(input_dim, target_length, learning_rate)
        
        self.cnn_filters = cnn_filters or [32, 64]
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 3]
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        
        # Build CNN feature extractor
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
        
        # LSTM takes the CNN output features
        self.lstm = nn.LSTM(in_channels, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, target_length)
        
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
                t20_weight=2.0,
                c50_weight=2.0,
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
        batch_size = x.shape[0]
        
        # CNN feature extraction
        # x shape: (batch_size, 1, input_dim)
        cnn_out = self.cnn_layers(x)  # shape: (batch_size, out_channels, input_dim)
        
        # Transpose for LSTM: (batch_size, seq_len, features)
        cnn_out = cnn_out.transpose(1, 2)  # shape: (batch_size, input_dim, out_channels)
        
        # LSTM forward
        _, (h_n, _) = self.lstm(cnn_out)  # h_n shape: (1, batch_size, lstm_hidden_dim)
        
        # Take last hidden state
        x = h_n[-1]  # shape: (batch_size, lstm_hidden_dim)
        
        # Dense layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # shape: (batch_size, target_length)
        
        return x


class CNNLSTMHybridV2(BaseEDCModel):
    """
    CNN-LSTM Hybrid Model V2: Parallel CNN and LSTM pathways.
    
    Architecture:
        Input ─→ CNN pathway ──┐
                               ├→ Concatenate → FC → Output
        Input ─→ LSTM pathway ─┘
    
    Two independent pathways processed in parallel, then merged.
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
        loss_type: str = "mse"
    ):
        """
        Initialize CNN-LSTM Hybrid V2 (Parallel).
        
        Args:
            input_dim: Number of input features
            target_length: Length of output EDC sequence
            cnn_filters: List of filter counts for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden_dim: Number of LSTM hidden units
            fc_hidden_dim: Number of fully connected hidden units
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function ('mse', 'edc_rir')
        """
        super().__init__(input_dim, target_length, learning_rate)
        
        self.cnn_filters = cnn_filters or [32, 64]
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 3]
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        
        # CNN pathway
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
        
        # LSTM pathway
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.lstm_fc = nn.Linear(lstm_hidden_dim, 256)
        
        # Merge and output
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, fc_hidden_dim)  # 256 + 256
        self.fc2 = nn.Linear(fc_hidden_dim, target_length)
        
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
                t20_weight=2.0,
                c50_weight=2.0,
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
        # CNN pathway
        cnn_out = self.cnn_layers(x)  # shape: (batch_size, channels, input_dim)
        cnn_out = self.cnn_pool(cnn_out)  # shape: (batch_size, channels, 1)
        cnn_out = cnn_out.view(cnn_out.shape[0], -1)  # shape: (batch_size, channels)
        cnn_out = torch.relu(self.cnn_fc(cnn_out))  # shape: (batch_size, 256)
        
        # LSTM pathway - x shape: (batch_size, 1, input_dim) - already correct format
        lstm_out, (h_n, _) = self.lstm(x)  # LSTM processes sequence
        lstm_out = h_n[-1]  # shape: (batch_size, lstm_hidden_dim)
        lstm_out = torch.relu(self.lstm_fc(lstm_out))  # shape: (batch_size, 256)
        
        # Merge pathways
        merged = torch.cat([cnn_out, lstm_out], dim=1)  # shape: (batch_size, 512)
        
        # Output layers
        x = torch.relu(self.fc1(merged))
        x = self.dropout(x)
        x = self.fc2(x)  # shape: (batch_size, target_length)
        
        return x


class CNNLSTMHybridV3(BaseEDCModel):
    """
    CNN-LSTM Hybrid Model V3: Multi-scale CNN-LSTM.
    
    Architecture:
        Input → Multi-scale CNN (different kernel sizes) → Concatenate → LSTM → FC → Output
    
    Multiple CNN pathways with different receptive fields process the input in parallel,
    capturing features at different scales.
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
        Initialize CNN-LSTM Hybrid V3 (Multi-scale).
        
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
        
        # Multi-scale CNN pathways
        self.conv_kernels = [3, 5, 7]
        self.conv_layers = nn.ModuleList()
        
        for kernel_size in self.conv_kernels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(32),
                    nn.ReLU()
                )
            )
        
        # LSTM (input size = 32 * num_scales)
        lstm_input_dim = 32 * len(self.conv_kernels)
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, target_length)
        
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
                t20_weight=2.0,
                c50_weight=2.0,
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
        # Multi-scale CNN
        multi_scale_features = []
        for conv in self.conv_layers:
            features = conv(x)  # shape: (batch_size, 32, input_dim)
            multi_scale_features.append(features)
        
        # Concatenate along feature dimension
        cnn_out = torch.cat(multi_scale_features, dim=1)  # shape: (batch_size, 96, input_dim)
        
        # Transpose for LSTM: (batch_size, seq_len, features)
        cnn_out = cnn_out.transpose(1, 2)  # shape: (batch_size, input_dim, 96)
        
        # LSTM forward
        _, (h_n, _) = self.lstm(cnn_out)  # h_n shape: (1, batch_size, lstm_hidden_dim)
        
        # Take last hidden state
        x = h_n[-1]  # shape: (batch_size, lstm_hidden_dim)
        
        # Dense layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # shape: (batch_size, target_length)
        
        return x
