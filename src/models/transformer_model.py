"""
Transformer-based model for EDC prediction.
"""

import torch
import torch.nn as nn
from .lstm_model import EDCRIRLoss
from .base_model import BaseEDCModel


class TransformerModel(BaseEDCModel):
    """
    Transformer-based model for EDC prediction from room features.
    
    Architecture:
        Input → Linear embedding → Positional encoding → 
        Transformer encoder → Linear → Output
    """
    
    def __init__(
        self,
        input_dim: int,
        target_length: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        loss_type: str = "mse"
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features
            target_length: Length of output EDC sequence
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function ('mse', 'edc_rir')
        """
        super().__init__(input_dim, target_length, learning_rate)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.fc_head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, target_length)
        )
        
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
        # Squeeze the sequence dimension: (batch_size, 1, input_dim) → (batch_size, input_dim)
        x = x.squeeze(1)  # shape: (batch_size, input_dim)
        
        # Embed input
        x = self.input_embed(x)  # shape: (batch_size, embed_dim)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # shape: (batch_size, 1, embed_dim)
        
        # Apply positional encoding
        x = self.positional_encoding(x)  # shape: (batch_size, 1, embed_dim)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # shape: (batch_size, 1, embed_dim)
        
        # Take mean over sequence dimension or use [CLS] token approach
        x = x.mean(dim=1)  # shape: (batch_size, embed_dim)
        
        # Output head
        x = self.fc_head(x)  # shape: (batch_size, target_length)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.tensor(10000.0).log() / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
