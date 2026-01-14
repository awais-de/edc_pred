# 📋 Getting Started - Step-by-Step Guide

## Overview

You've been provided with a complete development framework for training and comparing deep learning models for Energy Decay Curve (EDC) prediction. This guide walks you through the exact steps to begin.

## ✅ What Was Set Up

1. **Modular code structure** (`src/` directory) with:
   - Model architectures (LSTM + 3 CNN-LSTM hybrids)
   - Data loading and preprocessing utilities
   - Evaluation metrics (EDC + acoustic parameters)
   - Training infrastructure with PyTorch Lightning

2. **Documentation**:
   - DEVELOPMENT_ROADMAP.md - 6-phase plan
   - QUICKSTART.md - Quick reference
   - SETUP_COMPLETE.md - Overview of everything
   - RESULTS_TEMPLATE.md - For tracking experiments
   - This file - Getting started guide

3. **Training infrastructure**:
   - train_model.py - Full training script
   - Support for multiple architectures
   - Automatic logging and checkpointing

## 🚀 Your First Steps (in order)

### Step 1: Verify the Setup (5 minutes)

Check that all files were created:

```bash
cd /Users/muhammadawais/Downloads/ADSP/proj/edc_pred

# Verify structure
ls -la src/models/        # Should show: base_model.py, lstm_model.py, hybrid_models.py, __init__.py
ls -la src/data/          # Should show: data_loader.py, __init__.py
ls -la src/evaluation/    # Should show: metrics.py, __init__.py

# Verify new scripts
ls train_model.py         # Main training script
```

### Step 2: Test with Small Dataset (10-15 minutes)

Start with a quick test to verify everything works:

```bash
cd /Users/muhammadawais/Downloads/ADSP/proj/edc_pred

# Test LSTM baseline on 300 samples
python train_model.py \
  --model lstm \
  --edc-path data/raw/EDC \
  --features-path data/raw/roomFeaturesDataset.csv \
  --max-samples 300 \
  --batch-size 8 \
  --max-epochs 5 \
  --learning-rate 0.001 \
  --output-dir experiments
```

**What to expect:**
- Loading message showing 300 EDC files loaded
- Training progress bar for 5 epochs
- Results saved to `experiments/lstm_YYYYMMDD_HHMMSS/`
- Check files:
  - `metadata.json` - Training config and final metrics
  - `predictions.npy` - Model predictions
  - `targets.npy` - Ground truth
  - `tensorboard_logs/` - For visualization

### Step 3: Compare Architectures (30-45 minutes)

Now run the same test with the hybrid models:

```bash
# Test Hybrid V1 (CNN→LSTM sequential)
python train_model.py --model hybrid_v1 --max-samples 300 --max-epochs 5

# Test Hybrid V2 (Parallel CNN+LSTM)
python train_model.py --model hybrid_v2 --max-samples 300 --max-epochs 5

# Test Hybrid V3 (Multi-scale CNN)
python train_model.py --model hybrid_v3 --max-samples 300 --max-epochs 5
```

### Step 4: Analyze Results (10 minutes)

Compare the four models using the metadata files:

```bash
# View results for all experiments
cd experiments

# Check latest results
ls -lt | head -10  # See newest experiments

# View metrics for LSTM
cat lstm_*/metadata.json | grep -E '"mae"|"rmse"|"r2"' | head -20

# Compare MAE values across models (manual inspection of each metadata.json)
```

### Step 5: Document Baseline (5 minutes)

Copy a filled-in template to track results:

```bash
cp RESULTS_TEMPLATE.md BASELINE_RESULTS.md

# Edit BASELINE_RESULTS.md with your results
# Fill in the "Experiment Summary" table with your 4 runs
```

## 📊 Understanding the Output

After each training run, you'll get:

```
experiments/
└── lstm_20250110_143022/              # Timestamp-based folder
    ├── metadata.json                  # Training config + metrics
    ├── predictions.npy                # Model predictions (rescaled)
    ├── targets.npy                    # Ground truth (rescaled)
    ├── scaler_X.pkl                   # Input scaler (for inference)
    ├── scaler_y.pkl                   # Output scaler (for inference)
    ├── checkpoints/
    │   └── best_model.ckpt           # Best model checkpoint
    └── tensorboard_logs/              # For TensorBoard visualization
```

### Viewing TensorBoard Results

```bash
# From project root
tensorboard --logdir experiments/

# Then open browser to http://localhost:6006
```

## 🎯 Key Architecture Differences

To understand which to use:

- **LSTM**: Pure baseline, simplest architecture
- **Hybrid-v1**: CNN extracts features, LSTM generates sequence (most intuitive)
- **Hybrid-v2**: Two parallel pathways, might capture complementary patterns
- **Hybrid-v3**: Multi-scale CNN, should capture patterns at different scales (most sophisticated)

## 📈 Success Indicators

After Step 2, you should see:
- ✅ Data loads successfully
- ✅ Model trains without errors
- ✅ Validation loss decreases over epochs
- ✅ Results saved with metrics

After Step 3, you should be able to:
- ✅ Compare MAE/RMSE/R² across models
- ✅ Identify which architecture performs best
- ✅ See training/validation loss curves in TensorBoard

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you're in the right directory and src/ is properly structured.

```bash
# Verify you're in the right place
pwd  # Should end with /edc_pred

# Verify imports work
python -c "from src.models import get_model; print('OK')"
```

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size or max-samples:

```bash
python train_model.py --model lstm --max-samples 100 --batch-size 4 --max-epochs 5
```

### Issue: Training is very slow
**Solution**: Check if GPU is being used:

```bash
# Use CPU for first test if needed
python train_model.py --model lstm --device cpu --max-samples 300 --max-epochs 5
```

### Issue: Data not found
**Solution**: Verify file paths:

```bash
# Check if files exist
ls data/raw/EDC/*.npy | head -5         # Should show EDC files
cat data/raw/roomFeaturesDataset.csv | head -3  # Should show CSV content
```

## 📚 Next Phase: Scaling Up

Once you've verified the setup with 300 samples, scale up:

```bash
# Medium dataset (faster iteration)
python train_model.py --model lstm --max-samples 1000 --max-epochs 50

# Larger dataset (better results)
python train_model.py --model lstm --max-samples 4000 --max-epochs 100

# Full dataset (best results, takes longer)
python train_model.py --model lstm --max-samples 17640 --max-epochs 200
```

## 📋 Typical Workflow

Here's a suggested workflow for your project:

### Week 1: Verify & Baseline
- [x] Run 300-sample test on all architectures
- [ ] Run 1000-sample test on all architectures
- [ ] Document baseline metrics
- [ ] Identify best-performing architecture

### Week 2: Optimization
- [ ] Try different learning rates (1e-2, 1e-3, 1e-4)
- [ ] Try different batch sizes (4, 8, 16, 32)
- [ ] Run on full 6000-sample dataset
- [ ] Analyze error patterns

### Week 3-4: Enhancement
- [ ] Implement data augmentation
- [ ] Test feature engineering improvements
- [ ] Refine best architecture
- [ ] Prepare final results

### Week 5-6: Documentation
- [ ] Write methodology report
- [ ] Prepare comparison tables
- [ ] Document findings
- [ ] Submit final deliverables

## 💾 Important: Saving Your Work

After each run, save the results:

```bash
# Create an experiments log
date >> experiments/LOG.txt
echo "Model: lstm, Samples: 300, MAE: 0.XXXX" >> experiments/LOG.txt

# Or better, maintain BASELINE_RESULTS.md
# with a table of all experiments
```

## 🎓 Learning Resources

As you work through this, review:

1. **DEVELOPMENT_ROADMAP.md** - Understanding the phases
2. **QUICKSTART.md** - Code examples
3. **src/models/__init__.py** - How model registry works
4. **src/data/data_loader.py** - Data pipeline details
5. **src/evaluation/metrics.py** - Metric computation

## ✨ Tips for Success

1. **Start small**: Always test with 300 samples first
2. **Monitor GPU**: Use `nvidia-smi` if on NVIDIA GPU
3. **Save experiments**: Every run gets its own folder with timestamp
4. **Compare fairly**: Keep data splits consistent across runs
5. **Document results**: Update BASELINE_RESULTS.md after each run
6. **Version control**: Commit working code regularly

## 🚦 Ready to Start?

Once you've completed Step 1 (verification), you're ready to run:

```bash
python train_model.py --model lstm --max-samples 300 --max-epochs 5
```

This single command will:
1. Load 300 EDC files and room features
2. Scale the data
3. Create train/val/test splits
4. Initialize LSTM model
5. Train for up to 5 epochs (with early stopping)
6. Evaluate on test set
7. Save all results with metadata

**Estimated time**: 3-5 minutes

---

**Questions?** Refer to QUICKSTART.md or DEVELOPMENT_ROADMAP.md

**Ready?** Run the command above and check back in 5 minutes! ✅
