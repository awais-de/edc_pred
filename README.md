# 🎵 EDC Prediction: Deep Learning for Room Acoustics

Generalized prediction of **Energy Decay Curves (EDCs)** from room geometry using deep neural networks.

## 🚀 Quick Start

```bash
# First training run (5-10 minutes)
python train_model.py --model lstm --max-samples 300 --max-epochs 5

# Compare all architectures
for m in lstm hybrid_v1 hybrid_v2 hybrid_v3; do
  python train_model.py --model $m --max-samples 300 --max-epochs 5
done
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | Complete overview (START HERE) |
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Step-by-step quick start |
| **[DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)** | 6-phase development plan |
| **[QUICKSTART.md](QUICKSTART.md)** | Code examples & usage |
| **[FAQ_TROUBLESHOOTING.md](FAQ_TROUBLESHOOTING.md)** | Common issues & solutions |
| **[RESULTS_TEMPLATE.md](RESULTS_TEMPLATE.md)** | Track experiments |
| **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** | What was created |

## 🏗️ Architecture

Four complementary model architectures for comparison:

- **LSTM**: Pure LSTM baseline
- **Hybrid-v1**: Sequential CNN→LSTM
- **Hybrid-v2**: Parallel CNN+LSTM pathways  
- **Hybrid-v3**: Multi-scale CNN→LSTM

## 📊 Project Structure

```
src/
├── models/              # Model implementations
├── data/               # Data utilities
├── evaluation/         # Metrics & evaluation
├── training/           # Training utilities
├── configs/            # Configuration files
└── utils/              # Helper utilities
```

## 🎯 Evaluation Targets

| Metric | MAE | RMSE | R² |
|--------|-----|------|-----|
| EDT (s) | 0.020 | 0.02 | 0.98 |
| T20 (s) | 0.020 | 0.03 | 0.98 |
| C50 (dB) | 0.90 | 2 | 0.98 |

## ⚡ Key Features

✅ Modular architecture for easy model comparison  
✅ Automatic data loading, scaling, and splitting  
✅ Comprehensive evaluation metrics  
✅ PyTorch Lightning integration  
✅ Automatic checkpointing and logging  
✅ Extensive documentation & examples  
✅ Production-ready code  

## 🔧 Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- scikit-learn, numpy, pandas
- All packages in `requirements.txt`

## 📖 Where to Start

1. **First time?** → Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. **Ready to code?** → Read [GETTING_STARTED.md](GETTING_STARTED.md)
3. **Need examples?** → Check [QUICKSTART.md](QUICKSTART.md)
4. **Got errors?** → See [FAQ_TROUBLESHOOTING.md](FAQ_TROUBLESHOOTING.md)
5. **Planning phases?** → Review [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)

## 📝 Dataset

- **17,640 rooms** with 30 absorption cases each
- **Room features**: Geometry, positions, absorption coefficients
- **Targets**: Energy Decay Curves (96,000 samples each)
- **Source**: https://github.com/TUIlmenauAMS/LSTM-Model-Energy-Decay-Curves

## 🎓 Project Goals

Design and train new deep learning architectures to improve EDC prediction from room geometry, with evaluation on acoustic parameters (EDT, T20, C50).

## 📞 Questions?

Every question is answered in the documentation. Start with [GETTING_STARTED.md](GETTING_STARTED.md) or check [FAQ_TROUBLESHOOTING.md](FAQ_TROUBLESHOOTING.md).

---

**Ready to begin?** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for complete setup details.
