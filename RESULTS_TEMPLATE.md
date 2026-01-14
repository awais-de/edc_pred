# Model Comparison & Results Template

## Purpose
This document is for tracking and comparing results across different model architectures and configurations.

## Experiment Summary

| Experiment ID | Model | Dataset Size | Batch Size | Learning Rate | Max Epochs | Status |
|---|---|---|---|---|---|---|
| EXP-001 | LSTM | 600 | 8 | 0.001 | 50 | Pending |
| EXP-002 | Hybrid-v1 | 600 | 8 | 0.001 | 50 | Pending |
| EXP-003 | Hybrid-v2 | 600 | 8 | 0.001 | 50 | Pending |
| EXP-004 | Hybrid-v3 | 600 | 8 | 0.001 | 50 | Pending |

## Detailed Results

### Experiment Template

**ID**: EXP-XXX  
**Model**: [Architecture name]  
**Date**: [YYYY-MM-DD HH:MM:SS]  
**Dataset**: [num samples]  
**Output Directory**: `experiments/[model]_[timestamp]/`

#### Configuration
```yaml
model:
  name: [lstm/hybrid_v1/hybrid_v2/hybrid_v3]
  parameters: [list key params]
  
training:
  batch_size: [number]
  learning_rate: [value]
  max_epochs: [number]
  actual_epochs: [number trained before stopping]
  training_duration: [minutes]
```

#### Results

**Overall EDC Metrics:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE | [value] | - | ✓/✗ |
| RMSE | [value] | - | ✓/✗ |
| R² | [value] | - | ✓/✗ |

**EDT Metrics:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE (s) | [value] | 0.020 | ✓/✗ |
| RMSE (s) | [value] | 0.020 | ✓/✗ |
| R² | [value] | 0.98 | ✓/✗ |

**T20 Metrics:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE (s) | [value] | 0.020 | ✓/✗ |
| RMSE (s) | [value] | 0.030 | ✓/✗ |
| R² | [value] | 0.98 | ✓/✗ |

**C50 Metrics:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE (dB) | [value] | 0.90 | ✓/✗ |
| RMSE (dB) | [value] | 2.00 | ✓/✗ |
| R² | [value] | 0.98 | ✓/✗ |

#### Observations
- [Key finding 1]
- [Key finding 2]
- [Comparison with other models]
- [Unexpected results]

#### Next Steps
- [ ] [Action item 1]
- [ ] [Action item 2]

---

## Comparison Summary

### Architecture Performance Ranking

(After all baselines are run)

| Rank | Model | Best Metric | Overall MAE | Training Time | Memory |
|------|-------|------------|-------------|---------------|--------|
| 1 | [best] | [metric] | [value] | [time] | [usage] |
| 2 | [second] | [metric] | [value] | [time] | [usage] |
| 3 | [third] | [metric] | [value] | [time] | [usage] |
| 4 | [fourth] | [metric] | [value] | [time] | [usage] |

### Key Findings

#### Best Architecture for:
- **Overall EDC prediction**: [model name]
- **EDT estimation**: [model name]
- **T20 estimation**: [model name]
- **C50 estimation**: [model name]
- **Training efficiency**: [model name]
- **Inference speed**: [model name]

#### Critical Insights
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

---

## Hyperparameter Sensitivity Analysis

### Learning Rate Impact
```
Model: [name]
Dataset: [size]

Learning Rate | Best Val Loss | Final MAE | Epochs to Converge
1.0e-2        | [value]       | [value]   | [number]
1.0e-3        | [value]       | [value]   | [number]
1.0e-4        | [value]       | [value]   | [number]
1.0e-5        | [value]       | [value]   | [number]
```

### Batch Size Impact
```
Batch Size | Training Time | Final MAE | Memory (GB)
8          | [time]        | [value]   | [usage]
16         | [time]        | [value]   | [usage]
32         | [time]        | [value]   | [usage]
64         | [time]        | [value]   | [usage]
```

---

## Data Scale Sensitivity

| Dataset Size | Best Model | MAE | Training Time | Notes |
|---|---|---|---|---|
| 300 | [model] | [value] | [time] | Quick iteration |
| 600 | [model] | [value] | [time] | Good baseline |
| 1500 | [model] | [value] | [time] | Reasonable time |
| 3000 | [model] | [value] | [time] | Longer training |
| 6000+ | [model] | [value] | [time] | Full dataset |

---

## Error Analysis

### Failure Cases
- [Type of error 1]: [cause]
- [Type of error 2]: [cause]
- [Type of error 3]: [cause]

### Room Property Correlations
- Rooms with large volume: [observation]
- Rooms with absorptive materials: [observation]
- Rooms with reflective surfaces: [observation]

---

## Recommendations

### For Production Use
- **Recommended Model**: [architecture]
- **Configuration**: [specific params]
- **Expected Performance**: [metrics]
- **Trade-offs**: [speed vs accuracy]

### For Further Improvement
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

### Data Augmentation Strategies to Try
- [ ] Temporal augmentation (stretching/shifting)
- [ ] Feature noise injection
- [ ] Room property interpolation
- [ ] Domain randomization

### Architecture Improvements
- [ ] Attention mechanisms
- [ ] Residual connections
- [ ] Feature fusion strategies
- [ ] Multi-task learning

---

## Notes & References

- **Original Baseline**: models/old/lstm_model_train.py
- **Training Script**: train_model.py
- **Development Guide**: DEVELOPMENT_ROADMAP.md
- **Dataset**: 17,640 rooms × 30 absorption cases
- **Evaluation Targets**: EDT (0.020s), T20 (0.020s), C50 (0.90dB)

---

**Last Updated**: [Date]  
**Prepared By**: [Your name]
