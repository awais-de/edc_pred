## INFERENCE GUIDE

Quick reference for using the EDC prediction model.

---

## ⚡ 5-Minute Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model exists
ls experiments/multihead_20260123_120009/checkpoints/best_model.ckpt

# 3. Run first inference
python inference.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 0
```

**Expected Output**:
```
============================================================
EDC PREDICTION INFERENCE
============================================================
✓ Loaded scaler from ...
✓ Model loaded on cuda

🔍 Predicting for room index 0...

✓ Prediction successful!
  EDC shape: (1, 96000)
  T20: 0.2345 s
  C50: 5.6789 dB

============================================================
✓ Inference completed successfully!
============================================================
```

---

## 🔍 Common Commands

### Single Room Prediction
```bash
python inference.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 42
```

### Batch Prediction (Multiple Rooms)
```bash
python inference.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --indices 0 1 2 3 4 5
```

### Full Evaluation & Metrics
```bash
python evaluate.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --edc-dir data/raw/EDC \
  --output my_results
```

### CPU Mode (No GPU)
```bash
python inference.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 0 \
  --device cpu
```

### Custom Output Directory
```bash
python evaluate.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --output /path/to/custom_results_dir
```

---

## 🐍 Python API Usage

### Basic Prediction
```python
from inference import EDCPredictor
import numpy as np
import pandas as pd

# Initialize
predictor = EDCPredictor(
    checkpoint_path="experiments/multihead_20260123_120009/checkpoints/best_model.ckpt",
    features_csv="data/raw/roomFeaturesDataset.csv"
)

# Load features
df = pd.read_csv("data/raw/roomFeaturesDataset.csv")
features = df.iloc[0].select_dtypes(float).values.reshape(1, -1)

# Make prediction
results = predictor.predict(features)
print(f"EDC shape: {results['edc'].shape}")
print(f"T20: {results['t20'][0]:.4f} s")
print(f"C50: {results['c50'][0]:.4f} dB")
```

### With Acoustic Parameters
```python
# Get acoustic parameters (EDT, T20, C50)
results = predictor.predict_and_analyze(features, sample_rate=48000)

edc_pred = results['edc_predictions'][0]
t20_pred = results['t20_predictions'][0]
c50_pred = results['c50_predictions'][0]

params = results['acoustic_parameters'][0]
print(f"EDT: {params['edt']:.4f} s")
print(f"T20: {t20_pred:.4f} s")
print(f"C50: {c50_pred:.4f} dB")
```

### Batch Predictions
```python
# Multiple rooms
features_batch = df.iloc[:10].select_dtypes(float).values  # 10 samples

results = predictor.predict(features_batch)
print(f"Predicted {len(results['t20'])} T20 values")
print(f"T20s: {results['t20']}")
print(f"C50s: {results['c50']}")
```

### With Numpy Arrays
```python
# Create custom features (must be 16-dimensional)
features = np.random.randn(1, 16)  # 1 sample, 16 features

# Features will be automatically normalized
results = predictor.predict(features, normalize=True)
```

---

## 📊 Understanding Output

### What You Get Back

```python
{
    'edc': array of shape (n_samples, 96000)  # Energy Decay Curves
    't20': array of shape (n_samples,)         # Reverberation time (seconds)
    'c50': array of shape (n_samples,)         # Clarity index (dB)
    'acoustic_parameters': [                   # From compute_acoustic_parameters()
        {
            'edt': float,  # Early Decay Time (seconds)
            't20': float,  # Reverberation Time (seconds)  
            'c50': float,  # Clarity Index (dB)
            'c80': float,  # Definition Index (dB) - optional
        },
        ...
    ]
}
```

### Example Output
```python
>>> results = predictor.predict_and_analyze(features)
>>> results['t20_predictions']
array([0.2345678, 0.3456789, 0.4567890])

>>> results['c50_predictions']
array([5.123, 4.567, 6.789])

>>> results['acoustic_parameters'][0]
{'edt': 0.1234, 't20': 0.2346, 'c50': 5.123}
```

---

## 🎯 Evaluation Results Format

When you run `evaluate.py`, you get:

### Console Output
```
============================================================
EVALUATION RESULTS
============================================================

Parameter        MAE          RMSE         R²           Status
------------------------------------------------------------
T20 (s)          0.064680     0.110565     0.952980     ✓ PASS
C50 (dB)         0.338458     0.610199     0.991654     ✓ PASS
EDT (s)          0.000257     0.002129     0.999490     ✓ PASS
EDC (norm)       0.000001     0.000001     0.999990     ✓ PASS
============================================================
```

### Files Generated
```
results/
├── metrics_table.csv          # Detailed metrics in CSV format
├── edc_samples.png            # 4 example EDC predictions
├── t20_scatter.png            # T20 prediction accuracy plot
├── c50_scatter.png            # C50 prediction accuracy plot
├── edt_scatter.png            # EDT prediction accuracy plot
└── error_distributions.png    # Error histograms
```

### Metrics Explanation
- **MAE**: Mean absolute error (lower is better)
- **RMSE**: Root mean squared error (lower is better)
- **R²**: Coefficient of determination (higher is better, max=1)

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
**Solution**: Run from project root directory
```bash
cd /path/to/edc_pred
python inference.py ...
```

### "FileNotFoundError: checkpoint.ckpt"
**Solution**: Verify checkpoint path
```bash
# Check what experiments exist
ls experiments/

# Full path should be
experiments/multihead_20260123_120009/checkpoints/best_model.ckpt
```

### "CUDA out of memory"
**Solution**: Use CPU instead
```bash
python inference.py --device cpu
```

### "No module named 'torch'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
# or specifically:
pip install torch pytorch-lightning
```

### Predictions look wrong / all zeros
**Solution**: Check feature normalization
```python
# Should be normalized automatically, but if needed:
from src.data.data_loader import scale_data
_, _, scaler = scale_data(df_features)
features_scaled = scaler.transform(features)
results = predictor.predict(features_scaled, normalize=False)
```

---

## 📈 Understanding Model Inputs

### Feature Dimensions (16 total)

The model expects exactly 16 features in this order:

| # | Feature | Type | Range | Notes |
|---|---------|------|-------|-------|
| 1 | Room Length | float | > 0 | meters |
| 2 | Room Width | float | > 0 | meters |
| 3 | Room Height | float | > 0 | meters |
| 4 | Source X | float | 0-L | position |
| 5 | Source Y | float | 0-W | position |
| 6 | Source Z | float | 0-H | position |
| 7 | Receiver X | float | 0-L | position |
| 8 | Receiver Y | float | 0-W | position |
| 9 | Receiver Z | float | 0-H | position |
| 10 | Absorption 125Hz | float | 0-1 | coeff |
| 11 | Absorption 250Hz | float | 0-1 | coeff |
| 12 | Absorption 500Hz | float | 0-1 | coeff |
| 13 | Absorption 1kHz | float | 0-1 | coeff |
| 14 | Absorption 2kHz | float | 0-1 | coeff |
| 15 | Absorption 4kHz | float | 0-1 | coeff |
| 16 | (Reserved) | float | varies | normalized |

**Note**: Features are automatically normalized before inference.

---

## 💡 Pro Tips

1. **Batch Processing**: Processing multiple rooms at once is faster
   ```python
   # ❌ Slow: 100 individual calls
   for i in range(100):
       predictor.predict(features[i:i+1])
   
   # ✅ Fast: One batch call
   predictor.predict(features[:100])
   ```

2. **CPU vs GPU**: GPU is ~10-50× faster for large batches
   ```bash
   # First run: ~2-3 seconds (GPU warmup)
   # Subsequent runs: ~0.1-0.2 seconds per sample
   
   # CPU mode is slower but works everywhere
   python inference.py --device cpu
   ```

3. **Reuse Predictor**: Initialize once, predict many times
   ```python
   # ✅ Good
   predictor = EDCPredictor(...)
   for i in range(1000):
       results = predictor.predict(features[i])
   
   # ❌ Bad (slow initialization)
   for i in range(1000):
       predictor = EDCPredictor(...)  # Don't do this!
       results = predictor.predict(features[i])
   ```

4. **Memory**: Full evaluation uses ~2-3 GB VRAM
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1  # Updates every second
   ```

5. **Seed Reproducibility**: Results are deterministic (no random dropout at inference)
   ```python
   # Same features → Same predictions (100% reproducible)
   results1 = predictor.predict(features)
   results2 = predictor.predict(features)
   # results1 == results2 (exactly)
   ```

---

## 📚 Next Steps

1. **Make predictions**: `python inference.py --index 0`
2. **Evaluate full model**: `python evaluate.py`
3. **Use in Python**: Import `EDCPredictor` class
4. **Integrate into application**: See Python API examples above
5. **Train your own**: See [README.md](README.md#training)

---

## 🎓 For Evaluators

**Fastest path to verify results** (3 minutes):

```bash
# Install
pip install -r requirements.txt

# Quick test (single room)
python inference.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt \
  --features data/raw/roomFeaturesDataset.csv \
  --index 0

# Full evaluation (5 minutes)
python evaluate.py \
  --checkpoint experiments/multihead_20260123_120009/checkpoints/best_model.ckpt
  
# View results
cat results/metrics_table.csv
```

**Expected Metrics**:
- EDT: MAE < 0.0003, RMSE < 0.003, R² > 0.99
- T20: MAE < 0.07, RMSE < 0.12, R² > 0.95
- C50: MAE < 0.35, RMSE < 0.62, R² > 0.99

---

**Questions?** Check [README.md](README.md#troubleshooting) or [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)
