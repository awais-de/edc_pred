# Energy Decay Curve Prediction from Room Geometry

Deep learning framework for predicting acoustic room characteristics from geometric and material properties. This project develops and trains multi-task neural networks to estimate Energy Decay Curves (EDCs) and derived acoustic parameters (Early Decay Time, Reverberation Time T20, Clarity Index C50) from room feature vectors.

## Project Overview

Acoustic simulation and room design typically require either expensive physical measurements or computationally intensive geometric acoustics simulations. This project addresses the inverse problem: predicting acoustic characteristics directly from architectural properties using supervised deep learning.

The framework implements multiple neural network architectures (CNN-LSTM multi-head networks, pure LSTM, Transformer variants) trained on 6,000 room configurations with corresponding EDCs computed through room acoustic simulation.

## Model Performance

The trained multi-head CNN-LSTM model achieves the following performance metrics on the test set:

| Acoustic Parameter | Mean Absolute Error | Root Mean Squared Error | R² Score |
|-------------------|---------------------|------------------------|----------|
| EDC (normalized)  | 0.000257            | 0.00213                | 0.9995   |
| T20 (seconds)     | 0.0647              | 0.1106                 | 0.9530   |
| C50 (decibels)    | 0.338               | 0.610                  | 0.9917   |

Model specifications:
- Architecture: Multi-head CNN-LSTM with shared fully-connected layers
- Input dimension: 16 room features
- Output dimension: 96,000 EDC samples + scalar T20, C50 predictions
- Parameters: 103,385,218 trainable parameters
- Training: 200 epochs, batch size 8, 94.2 minutes on single GPU

## Installation and Setup

### Requirements

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies are specified in `requirements.txt` and include:
- Core scientific computing: NumPy, Pandas, SciPy, scikit-learn
- Deep learning: PyTorch 2.0+, PyTorch Lightning 2.0+
- Audio processing: librosa, soundfile, pyroomacoustics
- Visualization: Matplotlib, TensorBoard
- Configuration: Hydra, OmegaConf

## Usage

### Inference: Making Predictions

Generate acoustic predictions for room configurations:

```bash
# Single room prediction
python inference.py \
  --checkpoint trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 0

# Multiple room predictions
python inference.py \
  --checkpoint trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --indices 0 1 2 3

# With visualization (generates EDC curve plots)
python inference.py \
  --checkpoint trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 0 \
  --visualize
```

Output files:
- EDC predictions and acoustic parameters printed to console
- Visualization plots saved to `edc_plots/` directory (when using `--visualize`)

### Model Evaluation

Comprehensive evaluation on test set:

```bash
python evaluate.py \
  --checkpoint trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --edc-dir data/raw/EDC
```

### Training

Train models from scratch or resume training:

```bash
# Train new model
python train_multihead.py \
  --max-samples 6000 \
  --batch-size 8 \
  --max-epochs 200

# Resume from checkpoint
python train_multihead.py \
  --max-samples 6000 \
  --checkpoint trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/latest.ckpt
```

## Directory Structure

```
edc_pred/
├── src/
│   ├── models/
│   │   ├── base_model.py            PyTorch Lightning base class
│   │   ├── multihead_model.py       Multi-head CNN-LSTM architecture
│   │   ├── lstm_model.py            LSTM baseline
│   │   ├── transformer_model.py     Transformer variant
│   │   └── hybrid_models.py         Alternative CNN-LSTM configurations
│   ├── data/
│   │   └── data_loader.py           Data preprocessing and loading
│   ├── evaluation/
│   │   └── metrics.py               Evaluation metrics and acoustic parameter computation
│   └── training/
│       └── __init__.py
│
├── data/
│   ├── raw/
│   │   ├── roomFeaturesDataset.csv  Input room geometry and material properties
│   │   └── EDC/                     Reference Energy Decay Curve files
│   ├── processed/                   Placeholder for processed data
│   └── external/                    Placeholder for external datasets
│
├── trained_models/
│   └── multihead_edc_baseline_v1_2026_01_23/
│       ├── checkpoints/
│       │   └── best_model.ckpt      Trained model weights (PyTorch Lightning checkpoint)
│       ├── metadata.json            Training configuration and performance metrics
│       ├── scaler_X.pkl             Feature normalization (StandardScaler)
│       ├── scaler_y.pkl             Target normalization (StandardScaler)
│       ├── tensorboard_logs/        Training visualization logs
│       ├── edc_predictions.npy      Model predictions (test set)
│       ├── edc_targets.npy          Reference EDC values (test set)
│       ├── t20_predictions.npy      T20 predictions (test set)
│       ├── t20_targets.npy          T20 references (test set)
│       ├── c50_predictions.npy      C50 predictions (test set)
│       └── c50_targets.npy          C50 references (test set)
│
├── edc_plots/                       Generated visualization outputs
├── scripts/                         Utility scripts for analysis and training
├── evaluate.py                      Evaluation script
├── inference.py                     Inference interface
├── train_multihead.py               Training script for multi-head model
├── requirements.txt                 Python package dependencies
└── README.md                        This file
```

## Input Data Specification

### Room Features (16 dimensions)

The model expects 16 room features as input:

1. Room length (m)
2. Room width (m)
3. Room height (m)
4. Wall absorption coefficient (4 values, one per pair of opposing walls + ceiling/floor)
5. Floor material type (categorical, encoded)
6. Ceiling material type (categorical, encoded)
7. Additional acoustic properties (variable)

Features must be normalized using the provided scaler (`scaler_X.pkl`) before inference.

### EDC Data

Energy Decay Curves are 96,000-sample arrays representing normalized acoustic energy decay. Computed via:

```
EDC[n] = 10 * log10(sum(h[k]^2 for k >= n) / sum(h[k]^2 for all k))
```

where h is the room impulse response.

## Model Architecture

The multi-head model combines two pathways:

1. CNN Pathway:
   - 1D convolutional layers: filters [32, 64], kernel size 3
   - BatchNormalization and ReLU activations
   - Processes feature vector as single-channel 1D signal

2. LSTM Pathway:
   - Bidirectional LSTM: 128 hidden units
   - Processes feature sequence with temporal context

3. Fusion:
   - Concatenated features passed to shared fully-connected layers
   - Architecture: FC(2048) -> ReLU -> Dropout(0.3) -> Output layers

4. Output Heads:
   - EDC head: 96,000 values (reconstructed decay curve)
   - T20 head: 1 scalar value (reverberation time at -20 dB)
   - C50 head: 1 scalar value (clarity index)

## Training Configuration

Hyperparameters used for the provided trained model:

- Optimizer: Adam with default parameters (lr=0.001)
- Loss function: Weighted multi-task loss
  - EDC loss weight: 1.0
  - T20 loss weight: 100.0
  - C50 loss weight: 50.0
- Gradient clipping: 1.0
- Batch size: 8
- Maximum epochs: 200
- Early stopping: Based on validation loss

## Output Interpretation

### Acoustic Parameters

The model predicts three standard acoustic parameters:

- **EDT (Early Decay Time)**: Time for energy to decay 10 dB in the early part of the impulse response. Measured in seconds.

- **T20 (Reverberation Time)**: Time for sound energy to decay 60 dB, estimated from the 20-30 dB decay range. Measured in seconds. Standard measure for room acoustics.

- **C50 (Clarity Index)**: Ratio of energy arriving in the first 50 ms to total energy. Measured in dB. Important for speech intelligibility.

### EDC Curve

The full EDC is a 96,000-sample array representing normalized acoustic energy decay over time. At 48 kHz sample rate, this corresponds to 2 seconds of temporal resolution.

## Visualization

With the `--visualize` flag, the inference script generates two-panel plots:

- Top panel: Full energy decay curve (0-2 seconds)
- Bottom panel: Early decay detail (0-100 ms)

Plots are saved to `edc_plots/` directory as PNG files (150 DPI).

## File Dependencies

Critical files required for inference:

1. Model checkpoint: `trained_models/multihead_edc_baseline_v1_2026_01_23/checkpoints/best_model.ckpt`
2. Feature scaler: `trained_models/multihead_edc_baseline_v1_2026_01_23/scaler_X.pkl`
3. Feature data: `data/raw/roomFeaturesDataset.csv`

The scaler file is critical: it ensures input features are normalized identically to training data. If absent, features are re-normalized from the CSV, which introduces data leakage and may degrade predictions.

## Reproducibility

To reproduce results:

1. Use provided trained model checkpoint (no retraining required)
2. Ensure Python version matches (3.8+)
3. Install exact dependency versions: `pip install -r requirements.txt`
4. Use provided feature and EDC data files unchanged
5. Run inference with provided checkpoint path

Training from scratch requires:
- Original raw data (6,000+ room configurations with ground truth EDCs)
- Sufficient GPU memory (16+ GB recommended for batch size 8)
- Equivalent training configuration (see metadata.json)

## Repository Structure

The repository has been organized to contain only essential project files:

- Training scripts and model definitions in `src/`
- Trained model checkpoint and metadata in `trained_models/`
- Data loading utilities and evaluation metrics
- Inference interface with visualization support
- Complete dependencies list in `requirements.txt`

The `.gitignore` file excludes:
- `edc_plots/`: Generated visualization outputs
- `trained_models/`: Large model checkpoint files (included in submission)
- Python cache and build artifacts
- Virtual environment directories

## Technical Notes

- The model is trained on simulated room acoustics data. Performance on real measured data may differ.
- EDC predictions are normalized. Denormalization requires the target scaler (`scaler_y.pkl`) for direct acoustic energy interpretation.
- GPU acceleration is recommended for training but not required for inference.
- The sklearn version warning during inference (if present) is harmless and does not affect accuracy.
- Feature normalization using `scaler_X.pkl` is critical for inference accuracy. The inference pipeline automatically loads this scaler.

---

For additional information, refer to docstrings and function signatures in the source code.
