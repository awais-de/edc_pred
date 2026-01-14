# EDC Prediction: LSTM-CNN Hybrid Development Roadmap

## Project Overview
Development of hybrid CNN-LSTM architecture for improved Energy Decay Curve (EDC) prediction from room features. Target is to exceed baseline LSTM performance on acoustic metrics (EDT, T20, C50).

## Phase 1: Project Setup & Infrastructure (Week 1)

### 1.1 Project Organization
- [x] Review existing LSTM baseline code
- [ ] Create modular project structure:
  ```
  src/
  ├── models/           # Model architectures
  ├── data/            # Data loading and preprocessing
  ├── training/        # Training loops and utilities
  ├── evaluation/      # Metrics and analysis
  ├── configs/         # Hydra configuration files
  └── utils/           # Helper functions
  ```
- [ ] Establish version control with clear commit messages
- [ ] Set up experiment tracking infrastructure

### 1.2 Configuration Management
- [ ] Create Hydra config structure for experiments
  - Model configurations (architecture, hyperparameters)
  - Training configurations (learning rate, batch size, epochs)
  - Data configurations (paths, preprocessing options)
  - Evaluation configurations (metrics, output directories)
- [ ] Create baseline configs for LSTM and hybrid architectures

### 1.3 Data Pipeline
- [ ] Create unified data loader module
- [ ] Implement data augmentation strategies:
  - Normalization variants (MinMax, StandardScaler, RobustScaler)
  - Temporal augmentation (time stretching, time shifting)
  - Feature scaling variants
- [ ] Add validation for data consistency

---

## Phase 2: Model Architecture Development (Week 2-3)

### 2.1 CNN-LSTM Hybrid Architecture Design
```
Input Features (Room Geometry, Absorption, Positions)
         ↓
    [CNN Layers]  ← Extract spatial/feature patterns
         ↓
   [Feature Maps]
         ↓
    [LSTM Layers]  ← Temporal sequence modeling
         ↓
   [Dense Layers]
         ↓
   EDC Predictions
```

**Proposed Architecture Variants:**
- **Hybrid-v1**: CNN→LSTM (sequential)
  - CNN for input feature extraction (1-2 Conv1d layers)
  - LSTM for sequence generation
  
- **Hybrid-v2**: Parallel CNN + LSTM
  - Parallel pathways merged before final layers
  
- **Hybrid-v3**: Multi-scale CNN-LSTM
  - Multiple CNN scales processing different feature aspects

### 2.2 Implementation Tasks
- [ ] Implement base `AbstractModel` class with common methods
- [ ] Create `LSTMModel` (refactored from existing code)
- [ ] Create `CNNLSTMHybrid` models (v1, v2, v3)
- [ ] Create `TransformerModel` (future alternative)
- [ ] Add model registry for easy switching

### 2.3 Loss Functions
- [ ] Implement custom acoustic loss functions:
  - EDC-RIR combined loss (existing)
  - EDT loss (early decay time)
  - T20 loss (reverberation time)
  - C50 loss (clarity index)
- [ ] Implement loss combination strategies

---

## Phase 3: Training Infrastructure (Week 3-4)

### 3.1 Training Loop Development
- [ ] Create unified trainer using PyTorch Lightning
- [ ] Implement experiment logging:
  - TensorBoard integration
  - Metrics tracking (train/val/test)
  - Model checkpointing strategy
  - Hyperparameter logging
- [ ] Add distributed training support (if needed)

### 3.2 Validation & Testing
- [ ] Implement train/val/test split strategies
- [ ] Create cross-validation utilities
- [ ] Add early stopping mechanisms
- [ ] Implement model evaluation callbacks

---

## Phase 4: Evaluation & Analysis (Week 4-5)

### 4.1 Metrics Implementation
- [ ] Implement project-specific metrics:
  - MAE, RMSE, R² (overall)
  - EDT metrics (MAE, RMSE, R²)
  - T20 metrics (MAE, RMSE, R²)
  - C50 metrics (MAE, RMSE, R²)
- [ ] Create acoustic parameter derivation from EDC:
  - Energy Decay Time (EDT)
  - Reverberation Time (T20)
  - Clarity index (C50)

### 4.2 Comparative Analysis
- [ ] Create comparison framework
- [ ] Generate performance reports:
  - Architecture comparison tables
  - Visualization of predictions vs ground truth
  - Error analysis by room size/properties
  - Learning curves
- [ ] Statistical significance testing

---

## Phase 5: Experimentation & Optimization (Week 5-6)

### 5.1 Baseline Experiments
- [ ] Run LSTM baseline (existing implementation)
  - Record baseline metrics
  - Establish reference performance
  
### 5.2 Hybrid Model Experiments
- [ ] Train Hybrid-v1 CNN-LSTM
- [ ] Train Hybrid-v2 Parallel CNN-LSTM
- [ ] Train Hybrid-v3 Multi-scale CNN-LSTM
- [ ] Compare architectures

### 5.3 Hyperparameter Optimization
- [ ] Implement grid search / random search
- [ ] Test learning rate schedules
- [ ] Optimize CNN filter sizes and depths
- [ ] Optimize LSTM hidden dimensions

### 5.4 Data Strategy Experiments
- [ ] Test different normalization strategies
- [ ] Implement & test data augmentation
- [ ] Test feature engineering variants:
  - Frequency-dependent features
  - Surface-area weighted features
  - Geometric ratios and derived features

---

## Phase 6: Documentation & Reporting (Week 6-7)

### 6.1 Code Documentation
- [ ] Add docstrings to all modules
- [ ] Create API documentation
- [ ] Write setup and usage guides

### 6.2 Experimental Report
- [ ] Write methodology section
- [ ] Create results section with tables/figures
- [ ] Write discussion on findings
- [ ] Document architectural choices and trade-offs
- [ ] Provide recommendations for future work

### 6.3 Reproducibility
- [ ] Create requirements.txt with pinned versions
- [ ] Document hardware used for training
- [ ] Create scripts for reproduction
- [ ] Add random seed management

---

## Evaluation Targets

| Metric | Target MAE | Target RMSE | Target R² |
|--------|-----------|-----------|----------|
| EDT (s) | 0.020 | 0.02 | 0.98 |
| T20 (s) | 0.020 | 0.03 | 0.98 |
| C50 (dB) | 0.90 | 2 | 0.98 |

---

## Key Milestones

1. **Milestone 1**: Project structure ready, data pipeline tested ✓ (Phase 1)
2. **Milestone 2**: Hybrid models implemented and baseline established (Phase 2-3)
3. **Milestone 3**: All experiments completed and results analyzed (Phase 4-5)
4. **Milestone 4**: Documentation complete and reproducibility verified (Phase 6)

---

## Architecture Decision Tree

```
Start
 │
 ├─→ Feature input (16D room features)
 │
 ├─→ CNN Stage: Extract feature patterns
 │    ├─ Conv1d(16 → 32 filters, kernel=3)
 │    ├─ BatchNorm + ReLU
 │    └─ Conv1d(32 → 64 filters, kernel=3)
 │
 ├─→ Reshape for LSTM
 │
 ├─→ LSTM Stage: Temporal modeling
 │    ├─ LSTM(64 → 128 hidden)
 │    ├─ Dropout(0.3)
 │    └─ LSTM(128 → 128 hidden)
 │
 ├─→ Dense layers:
 │    ├─ Linear(128 → 2048) + ReLU
 │    ├─ Dropout(0.3)
 │    └─ Linear(2048 → target_length)
 │
 └─→ EDC output (96000D sequence)
```

---

## Success Criteria

- [ ] Hybrid model outperforms baseline LSTM on at least one acoustic metric
- [ ] Code is modular, documented, and reproducible
- [ ] All experiments are logged and results are comparable
- [ ] Comprehensive report with findings and recommendations
- [ ] Clear path for future improvements identified

---

## Resources

- **Existing Code**: [models/old/lstm_model_train.py](models/old/lstm_model_train.py)
- **Dataset**: 17,640 rooms × 30 absorption cases
- **Reference**: https://github.com/TUIlmenauAMS/LSTM-Model-Energy-Decay-Curves
- **Libraries**: PyTorch, PyTorch Lightning, Hydra, scikit-learn

---

## Next Steps

1. Set up project structure and modules
2. Create configuration files
3. Refactor existing LSTM code into modules
4. Implement CNN-LSTM hybrid architecture
5. Create unified training infrastructure
6. Run baseline and hybrid experiments
7. Analyze and document results
