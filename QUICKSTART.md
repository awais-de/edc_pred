# EDC Prediction Development - Quick Start Guide

## 📋 Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- PyTorch Lightning
- All packages in `requirements.txt`

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
src/
├── models/              # Model architectures
│   ├── base_model.py   # Abstract base class
│   ├── lstm_model.py   # LSTM implementation
│   ├── hybrid_models.py # CNN-LSTM hybrids (v1, v2, v3)
│   └── __init__.py     # Model registry
├── data/               # Data utilities
│   ├── data_loader.py  # Loading and preprocessing
│   └── __init__.py
├── training/           # Training utilities
├── evaluation/         # Metrics and evaluation
│   ├── metrics.py      # Evaluation metrics
│   └── __init__.py
├── configs/            # Configuration files (YAML)
└── utils/             # Helper utilities
```

## 🚀 Quick Start: Train LSTM Baseline

```python
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.models import get_model
from src.data.data_loader import (
    load_edc_data, load_room_features, scale_data, create_dataloaders
)
from src.evaluation.metrics import evaluate_model, print_metrics

# 1. Load data
print("Loading data...")
edc_data = load_edc_data(
    "data/raw/EDC",
    target_length=96000,
    max_files=600  # Start with 600 files for quick testing
)
room_features = load_room_features("data/raw/roomFeaturesDataset.csv", max_samples=600)

# 2. Scale data
X_scaled, y_scaled, scaler_X, scaler_y = scale_data(
    room_features, edc_data, scaler_type="minmax"
)

# 3. Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    X_scaled, y_scaled, batch_size=8, train_ratio=0.6, val_ratio=0.2
)

# 4. Initialize model
model = get_model(
    "lstm",
    input_dim=16,
    target_length=96000,
    lstm_hidden_dim=128,
    fc_hidden_dim=2048,
    dropout_rate=0.3,
    learning_rate=0.001,
    loss_type="mse"
)

# 5. Train
trainer = Trainer(
    max_epochs=50,
    accelerator='auto',
    devices='auto',
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(monitor="val_loss", save_top_k=1)
    ]
)

trainer.fit(model, train_loader, val_loader)

# 6. Evaluate
model.eval()
preds, targets = [], []
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
metrics = evaluate_model(targets_rescaled, preds_rescaled)
print_metrics(metrics)
```

## 🔄 Train CNN-LSTM Hybrid

```python
# Same as above, but change model initialization:
model = get_model(
    "hybrid_v1",  # or "hybrid_v2", "hybrid_v3"
    input_dim=16,
    target_length=96000,
    cnn_filters=[32, 64],
    cnn_kernel_sizes=[3, 3],
    lstm_hidden_dim=128,
    fc_hidden_dim=2048,
    dropout_rate=0.3,
    learning_rate=0.001,
    loss_type="mse"
)
```

## 📊 Available Models

- **`lstm`**: Pure LSTM baseline
- **`hybrid_v1`**: Sequential CNN → LSTM
- **`hybrid_v2`**: Parallel CNN + LSTM pathways
- **`hybrid_v3`**: Multi-scale CNN → LSTM

## 🎯 Evaluation Targets

| Metric | MAE | RMSE | R² |
|--------|-----|------|-----|
| EDT (s) | 0.020 | 0.02 | 0.98 |
| T20 (s) | 0.020 | 0.03 | 0.98 |
| C50 (dB) | 0.90 | 2 | 0.98 |

## 📝 Configuration Files

Create experiment configs in `src/configs/`:

```yaml
# configs/lstm_baseline.yaml
model:
  name: lstm
  input_dim: 16
  target_length: 96000
  lstm_hidden_dim: 128
  fc_hidden_dim: 2048
  dropout_rate: 0.3
  learning_rate: 0.001
  loss_type: mse

training:
  max_epochs: 200
  batch_size: 8
  early_stopping_patience: 10

data:
  edc_path: data/raw/EDC
  features_path: data/raw/roomFeaturesDataset.csv
  target_length: 96000
  scaler_type: minmax
  max_samples: null
```

## 🔧 Next Steps

1. **Test with small dataset first** (few hundred samples)
2. **Compare architectures** on same train/val/test split
3. **Experiment with hyperparameters** (learning rate, batch size, etc.)
4. **Implement data augmentation** strategies
5. **Analyze errors** by room properties
6. **Document findings** in reports

## 📚 References

- Baseline code: [models/old/lstm_model_train.py](models/old/lstm_model_train.py)
- Dataset: [https://github.com/TUIlmenauAMS/LSTM-Model-Energy-Decay-Curves](https://github.com/TUIlmenauAMS/LSTM-Model-Energy-Decay-Curves)
- PyTorch Lightning: [https://lightning.ai/](https://lightning.ai/)

## ⚠️ Notes

- Start with smaller dataset (max_files=300-600) for quick iteration
- Monitor GPU/CPU memory usage
- Save checkpoints regularly
- Keep experiment logs for reproducibility
- Use TensorBoard for training visualization
