"""
Base model class for all EDC prediction architectures.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseEDCModel(pl.LightningModule, ABC):
    """
    Abstract base class for all EDC prediction models.
    
    All EDC models should inherit from this class and implement
    the forward() method specific to their architecture.
    """
    
    def __init__(self, input_dim: int, target_length: int, learning_rate: float = 0.001):
        """
        Initialize base model.
        
        Args:
            input_dim: Number of input features (room features)
            target_length: Length of output EDC sequence
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.input_dim = input_dim
        self.target_length = target_length
        self.learning_rate = learning_rate
        
        self.save_hyperparameters()
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - must be implemented by subclasses.
        
        Args:
            x: Input tensor of shape (batch_size, 1, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, target_length)
        """
        pass
    
    @abstractmethod
    def _build_criterion(self):
        """Build and set the loss function."""
        pass
    
    def training_step(self, batch, batch_idx):
        """Single training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        mae = torch.mean(torch.abs(y_hat - y))
        self.train_losses.append(loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        val_mae = torch.mean(torch.abs(y_hat - y))
        self.val_losses.append(loss.item())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', val_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch."""
        if self.train_losses:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            self.log('train_loss_epoch', avg_loss, prog_bar=True)
            self.train_losses.clear()
    
    def on_validation_epoch_end(self):
        """End of validation epoch."""
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log('val_loss_epoch', avg_loss, prog_bar=True)
            self.val_losses.clear()
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
