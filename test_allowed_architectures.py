#!/usr/bin/env python
"""
Test script for allowed architectures: hybrid_v1, hybrid_v2, transformer.
Runs each model on a small dataset to verify no errors.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Test configurations
TESTS = [
    {
        "name": "Hybrid V1 - Sequential CNN-LSTM",
        "model": "hybrid_v1",
        "samples": 300,
        "epochs": 5,
        "batch_size": 8
    },
    {
        "name": "Hybrid V2 - Parallel CNN-LSTM",
        "model": "hybrid_v2",
        "samples": 300,
        "epochs": 5,
        "batch_size": 8
    },
    {
        "name": "Transformer - Attention-based",
        "model": "transformer",
        "samples": 300,
        "epochs": 5,
        "batch_size": 8
    }
]

def run_test(test_config):
    """Run a single architecture test."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_config['name']}")
    print(f"{'='*70}")
    
    cmd = [
        "python", "train_model.py",
        "--model", test_config["model"],
        "--max-samples", str(test_config["samples"]),
        "--max-epochs", str(test_config["epochs"]),
        "--batch-size", str(test_config["batch_size"])
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"❌ FAILED")
            print(f"STDERR:\n{result.stderr}")
            return False
        else:
            print(f"✅ PASSED")
            # Extract key metrics from output
            if "Overall" in result.stderr or "Overall" in result.stdout:
                print("Output excerpt:")
                for line in result.stdout.split('\n')[-20:]:
                    if line.strip():
                        print(f"  {line}")
            return True
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (>600s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING ALLOWED ARCHITECTURES")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {}
    for test in TESTS:
        results[test["model"]] = run_test(test)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All architectures working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} architecture(s) need fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
