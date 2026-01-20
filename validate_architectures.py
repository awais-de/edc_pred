#!/usr/bin/env python
"""
Quick validation that all allowed architectures can be imported and instantiated.
"""

import sys
import torch

def validate_imports():
    """Verify all models can be imported."""
    print("Validating model imports...")
    
    try:
        from src.models import get_model, MODEL_REGISTRY
        print(f"✅ Model registry loaded")
        print(f"   Available models: {list(MODEL_REGISTRY.keys())}")
        
        # Check all models in registry
        for model_name in MODEL_REGISTRY.keys():
            model_class = MODEL_REGISTRY[model_name]
            print(f"✅ {model_name}: {model_class.__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def validate_instantiation():
    """Verify all allowed models can be instantiated."""
    print("\nValidating model instantiation...")
    
    from src.models import get_model
    
    allowed_models = ["hybrid_v1", "hybrid_v2", "transformer"]
    success_count = 0
    
    for model_name in allowed_models:
        try:
            model = get_model(
                model_name,
                input_dim=16,
                target_length=96000,
                learning_rate=0.001
            )
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {model_name}: {params:,} parameters")
            success_count += 1
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    return success_count == len(allowed_models)

def validate_forward_pass():
    """Verify models can process sample data."""
    print("\nValidating forward pass...")
    
    from src.models import get_model
    
    allowed_models = ["hybrid_v1", "hybrid_v2", "transformer"]
    success_count = 0
    
    # Create sample input
    batch_size = 4
    input_data = torch.randn(batch_size, 1, 16)  # (batch, 1, features)
    
    for model_name in allowed_models:
        try:
            model = get_model(
                model_name,
                input_dim=16,
                target_length=96000,
                learning_rate=0.001
            )
            model.eval()
            
            with torch.no_grad():
                output = model(input_data)
            
            assert output.shape == (batch_size, 96000), f"Wrong output shape: {output.shape}"
            print(f"✅ {model_name}: output shape {output.shape}")
            success_count += 1
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    return success_count == len(allowed_models)

def main():
    """Run all validations."""
    print("="*70)
    print("ALLOWED ARCHITECTURES VALIDATION")
    print("="*70)
    
    results = [
        ("Imports", validate_imports()),
        ("Instantiation", validate_instantiation()),
        ("Forward Pass", validate_forward_pass()),
    ]
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All validations passed! Ready for training.")
        print("\nRun training with:")
        print("  python train_model.py --model hybrid_v1 --max-samples 400")
        print("  python train_model.py --model hybrid_v2 --max-samples 400")
        print("  python train_model.py --model transformer --max-samples 400")
    else:
        print("⚠️  Some validations failed. See errors above.")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
