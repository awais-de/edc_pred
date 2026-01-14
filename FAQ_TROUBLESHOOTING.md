# ❓ FAQ & Troubleshooting Guide

## Common Questions

### Q1: Which model should I start with?
**A**: Start with **LSTM** for your baseline. It's simpler and will give you a reference point to compare hybrid models against.

```bash
python train_model.py --model lstm --max-samples 300 --max-epochs 5
```

### Q2: How do I know if it's working?
**A**: You should see:
1. Files being loaded (print statements showing file count)
2. Training progress bar advancing
3. Val loss values printing
4. A new folder created in `experiments/`

If you see errors, check Q6-Q10 below.

### Q3: What do the different hybrid models do?

| Model | Design | Use Case |
|-------|--------|----------|
| **LSTM** | Classic LSTM baseline | Reference point |
| **Hybrid-v1** | CNN then LSTM | Best if CNN features help LSTM |
| **Hybrid-v2** | CNN + LSTM in parallel | Combines different feature extraction paths |
| **Hybrid-v3** | Multi-scale CNN then LSTM | Better for multi-scale feature extraction |

**Recommendation**: Try all four on 300 samples, then focus on the best 1-2.

### Q4: How long should training take?

| Dataset | Batch Size | Model | Approximate Time |
|---------|-----------|-------|-----------------|
| 300 | 8 | LSTM | 2-5 minutes |
| 300 | 8 | Hybrid-v1 | 3-7 minutes |
| 600 | 8 | LSTM | 4-10 minutes |
| 1000 | 8 | LSTM | 8-15 minutes |
| 4000 | 16 | LSTM | 30-60 minutes |

**Times vary** based on your hardware (CPU vs GPU).

### Q5: What's the difference between MAE, RMSE, and R²?

- **MAE** (Mean Absolute Error): Average absolute difference. Lower is better. Same units as output.
- **RMSE** (Root Mean Squared Error): Penalizes large errors more. Lower is better.
- **R²** (Coefficient of Determination): Explains variance (0-1). Higher is better. 1.0 = perfect prediction.

For EDC: Focus on MAE and RMSE (in original units). R² should be > 0.9.

### Q6: Why is validation loss increasing (overfitting)?

**Causes:**
- Model too complex for the data
- Learning rate too high
- Not enough training data

**Solutions:**
```bash
# Try higher dropout
# Modify the model (reduce layers/parameters)
# Lower learning rate
python train_model.py --learning-rate 0.0001 ...

# Use more data
python train_model.py --max-samples 1000 ...
```

### Q7: How do I use my results for future inference?

Saved files include scalers. For inference on new data:

```python
import numpy as np
import torch
import joblib
from src.models import get_model

# Load trained model and scalers
model = get_model.load_from_checkpoint("path/to/best_model.ckpt")
scaler_X = joblib.load("experiments/lstm_.../scaler_X.pkl")
scaler_y = joblib.load("experiments/lstm_.../scaler_y.pkl")

# Prepare new room features
new_features = np.array([[...]])  # Shape: (1, 16)

# Scale input
X_scaled = scaler_X.transform(new_features)
X_scaled = X_scaled.reshape((-1, 1, 16))  # Add sequence dimension

# Predict
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    pred_scaled = model(X_tensor).numpy()

# Unscale output
pred_edc = scaler_y.inverse_transform(pred_scaled)
```

### Q8: Can I stop training early if it's taking too long?

**Yes!** Press `Ctrl+C` to stop. Your model will still save:
- The best checkpoint (before you stopped)
- The metrics up to that point

The results won't be as good, but you can still analyze what you have.

### Q9: What if I want to change the model architecture?

Edit the model definition in `src/models/hybrid_models.py` or `src/models/lstm_model.py`, then train:

```python
# Add your own model
class CustomModel(BaseEDCModel):
    def __init__(self, ...):
        super().__init__(...)
        # Your layers
    
    def forward(self, x):
        # Your forward pass
        return x
```

Then register it in `src/models/__init__.py`:

```python
MODEL_REGISTRY["my_custom"] = CustomModel
```

### Q10: What if results don't meet the targets?

**Don't worry!** This is expected. Here's what to try:

1. **Use more data**:
   ```bash
   python train_model.py --max-samples 4000 --max-epochs 100
   ```

2. **Try different architectures**:
   ```bash
   for model in lstm hybrid_v1 hybrid_v2 hybrid_v3; do
     python train_model.py --model $model --max-samples 1000
   done
   ```

3. **Optimize hyperparameters**:
   - Try learning rates: 1e-2, 1e-3, 1e-4, 1e-5
   - Try batch sizes: 4, 8, 16, 32

4. **Investigate data**:
   - Check if certain room types are harder to predict
   - Analyze error patterns (which EDC features are hardest to predict?)

---

## Troubleshooting

### Error: "FileNotFoundError: data/raw/EDC"

**Problem**: Data path is wrong.

**Solution**: Check where your data actually is:
```bash
# See what's in your data directory
ls -la data/

# If structure is different, update paths:
python train_model.py \
  --edc-path /path/to/your/EDC/folder \
  --features-path /path/to/your/features.csv
```

### Error: "ModuleNotFoundError: No module named 'src'"

**Problem**: Python can't find the src module.

**Solution**: Make sure you're running from the project root:
```bash
cd /Users/muhammadawais/Downloads/ADSP/proj/edc_pred
python train_model.py ...
```

### Error: "RuntimeError: CUDA out of memory"

**Problem**: GPU memory exceeded.

**Solution**: 
```bash
# Reduce batch size
python train_model.py --batch-size 4 ...

# Or use CPU
python train_model.py --device cpu ...

# Or use less data
python train_model.py --max-samples 300 ...
```

### Error: "KeyError: 'ID'" when loading features

**Problem**: CSV might not have ID column. This is fine!

**Solution**: The code handles this. If you get this error, check your CSV:
```bash
head -1 data/raw/roomFeaturesDataset.csv  # See column names
```

The code automatically drops 'ID' if present.

### Error: "IndexError: index out of range" during data loading

**Problem**: EDC files might be empty or have unexpected format.

**Solution**: Check your EDC files:
```python
import numpy as np
import os

edc_folder = "data/raw/EDC"
for fname in os.listdir(edc_folder)[:10]:
    edc = np.load(os.path.join(edc_folder, fname))
    print(f"{fname}: shape={edc.shape}, min={edc.min():.4f}, max={edc.max():.4f}")
```

### Slow Training (Even on GPU)

**Problem**: Model not using GPU.

**Solution**: Verify GPU is being used:
```bash
# Check NVIDIA GPU
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time

# Check if PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If GPU is available but not used, it might be a PyTorch installation issue. Try:
```bash
pip install torch --upgrade
```

### Results Not Saving

**Problem**: Experiments folder empty or missing results.

**Solution**: Check the script ran completely:
```bash
# Look for errors at the end of output
tail -50 <training_output>

# Check if experiments folder was created
ls -la experiments/

# Check if there's a newer folder
ls -lt experiments/ | head -5
```

---

## Performance Tips

### To Speed Up Training

1. **Use GPU**:
   ```bash
   # Automatic (uses GPU if available)
   python train_model.py --device auto ...
   
   # Force GPU
   python train_model.py --device cuda ...
   ```

2. **Increase batch size** (if GPU memory allows):
   ```bash
   python train_model.py --batch-size 32 ...
   ```

3. **Reduce early stopping patience** (stop faster if no improvement):
   - Edit `train_model.py`, line ~150: change `patience=15` to `patience=5`

4. **Use fewer samples for iteration**:
   ```bash
   python train_model.py --max-samples 300 ...  # Quick test
   python train_model.py --max-samples 2000 ... # Medium test
   ```

### To Improve Results

1. **More data**:
   ```bash
   python train_model.py --max-samples 6000 ...
   ```

2. **Longer training**:
   ```bash
   python train_model.py --max-epochs 200 ...
   ```

3. **Different learning rate**:
   ```bash
   python train_model.py --learning-rate 0.0005 ...
   ```

4. **Experiment with models**:
   ```bash
   # Try all models on same data
   for m in lstm hybrid_v1 hybrid_v2 hybrid_v3; do
     python train_model.py --model $m --max-samples 1000
   done
   ```

---

## Debugging Tips

### Print Input/Output Shapes

Add this to your training code:

```python
# After creating model
x_sample, y_sample = next(iter(train_loader))
print(f"Input shape: {x_sample.shape}")
print(f"Output shape: {y_sample.shape}")

# Forward pass
y_pred = model(x_sample)
print(f"Prediction shape: {y_pred.shape}")
```

### Check Data Quality

```python
# After loading data
import numpy as np

print(f"EDC stats:")
print(f"  Min: {edc_data.min():.4f}")
print(f"  Max: {edc_data.max():.4f}")
print(f"  Mean: {edc_data.mean():.4f}")
print(f"  Std: {edc_data.std():.4f}")
print(f"  NaN count: {np.isnan(edc_data).sum()}")
```

### Monitor Training Live

Use TensorBoard while training:

```bash
# In another terminal
tensorboard --logdir experiments/ --port 6006

# Open http://localhost:6006 in browser
```

---

## When Things Go Wrong

1. **Check the error message carefully** - it usually tells you exactly what's wrong
2. **Google the error** - prefix with "pytorch" or "pytorch lightning"
3. **Simplify the test** - reduce dataset size to isolate the issue
4. **Check your data** - print shapes and values to verify they're correct
5. **Try the CPU** - if GPU issues, test on CPU first
6. **Revert changes** - if you modified code, go back to working version

---

## Getting Help

### Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Lightning**: https://lightning.ai/
- **Scikit-learn**: https://scikit-learn.org/
- **Project Code**: Read the docstrings in src/models/ and src/data/

### Debug Checklist

Before asking for help:

- [ ] Data loads successfully?
- [ ] Model initializes without error?
- [ ] Training starts and makes progress?
- [ ] At least one model trained successfully?
- [ ] Results folder has metadata.json?

---

**Still stuck?** Review DEVELOPMENT_ROADMAP.md or QUICKSTART.md for examples.

Good luck! 🚀
