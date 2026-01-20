# Architecture Fixes & Validation Plan

## Status: READY FOR TESTING

### Recent Fixes Applied

#### 1. ✅ Hybrid V2 LSTM Pathway Fix
**File:** `src/models/hybrid_models.py` (lines 222-225)
**Issue:** LSTM input shape mismatch
**Fix Applied:** 
- Removed unnecessary squeeze/unsqueeze operations
- Input is already in correct format: (batch_size, 1, input_dim)
- LSTM with batch_first=True expects exactly this format
- Changed from: `lstm_in = x.squeeze(1).unsqueeze(1); self.lstm(lstm_in)`
- Changed to: `self.lstm(x)` directly

#### 2. ✅ Hybrid V3 Parameter Handling
**File:** `train_model.py` (lines 143-159)
**Issue:** hybrid_v3 doesn't accept cnn_filters/cnn_kernel_sizes parameters
**Fix Applied:**
- Added conditional logic to detect "hybrid_v3" specifically
- Only passes lstm_hidden_dim, fc_hidden_dim, dropout_rate for hybrid_v3
- Other hybrids get full CNN parameters

#### 3. ✅ Transformer Registration
**File:** `src/models/__init__.py`
**Issue:** Transformer not available in model registry
**Fix Applied:**
- Imported TransformerModel from transformer_model.py
- Added "transformer" entry to MODEL_REGISTRY
- Ready to instantiate with: `get_model("transformer", ...)`

#### 4. ✅ Transformer Implementation
**File:** `src/models/transformer_model.py` (164 lines)
**Status:** Complete and ready for testing
**Architecture:**
- Input embedding layer
- Positional encoding (sinusoidal)
- 4-layer transformer encoder
- 8 attention heads
- Supports both MSE and EDC+RIR loss functions

---

## Allowed Architectures Summary

### 1. Hybrid V1 (Sequential CNN→LSTM)
- **Status:** ✅ Working (tested on 300-600 samples)
- **Parameters:** ~197M
- **Performance:** MAE 0.005978 (2nd best)
- **Characteristics:** CNN extracts features, LSTM models sequence

### 2. Hybrid V2 (Parallel CNN + LSTM)  
- **Status:** 🔄 FIXED, Ready for test
- **Parameters:** ~198M
- **Previous Issue:** LSTM shape mismatch (FIXED)
- **Expected Performance:** Better feature combination than V1

### 3. Transformer (Attention-based)
- **Status:** ✅ Implemented, Ready for test
- **Parameters:** ~198M (similar to hybrids)
- **Architecture:** 4-layer encoder, 8-head attention
- **Advantage:** No sequential processing needed, parallel computation

---

## Validation Tests Required

### Test 1: Hybrid V2 Validation
```bash
python train_model.py --model hybrid_v2 --max-samples 400 --max-epochs 5 --batch-size 8
```
Expected: Training completes without shape errors

### Test 2: Transformer Validation  
```bash
python train_model.py --model transformer --max-samples 400 --max-epochs 5 --batch-size 8
```
Expected: Training completes without errors

### Test 3: Comprehensive Comparison (600 samples, 10 epochs)
```bash
# Sequential tests
python train_model.py --model hybrid_v1 --max-samples 600 --max-epochs 10 --batch-size 8
python train_model.py --model hybrid_v2 --max-samples 600 --max-epochs 10 --batch-size 8
python train_model.py --model transformer --max-samples 600 --max-epochs 10 --batch-size 8
```
Expected: All complete without errors, generate metrics for comparison

---

## Next Steps After Validation

1. **Performance Comparison:** Analyze metrics (MAE, RMSE, R² for EDT, T20, C50)
2. **Best Architecture Selection:** Choose hybrid_v1, hybrid_v2, or transformer
3. **Hyperparameter Optimization:** Tune learning rate, batch size, epochs
4. **Scale Up:** Run on 2000-4000 samples for better convergence
5. **Target Achievement:** Focus on improving T20/C50 metrics (currently over target)

---

## Technical Notes

### Input/Output Shapes
- Input: (batch_size, 1, input_dim) where input_dim = 16 (room features)
- Output: (batch_size, target_length) where target_length = 96000 (EDC sequence)

### Known Constraints
- GPU Memory: 4 parallel models cause OOM, run sequentially
- T20/C50 Metrics: Currently 7.5× and 2.2× over target (data/training issue)
- EDT Metric: ✅ Already exceeds target at 0.0138 (target: 0.020)

### Loss Functions Available
1. **MSE:** Simple mean squared error
2. **EDC+RIR:** Combined loss with acoustic weighting (alpha=1.0, beta=0.5)

---

## Commands for Quick Testing

**Quick Test (200 samples, 3 epochs):**
```bash
python train_model.py --model hybrid_v2 --max-samples 200 --max-epochs 3
python train_model.py --model transformer --max-samples 200 --max-epochs 3
```

**Comprehensive Test (600 samples, 10 epochs):**
```bash
python test_allowed_architectures.py
```

**Single Model Extended (1000 samples, 50 epochs):**
```bash
python train_model.py --model hybrid_v1 --max-samples 1000 --max-epochs 50 --batch-size 8
```

