#!/usr/bin/env python3
"""
Quick GPU resource check for parallel training feasibility.
"""

import subprocess
import sys

print("=" * 60)
print("GPU RESOURCE CHECK FOR PARALLEL TRAINING")
print("=" * 60)
print()

try:
    # Get GPU info
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    
    lines = result.stdout.strip().split('\n')
    num_gpus = len(lines)
    
    print(f"Found {num_gpus} GPU(s):")
    print()
    
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(',')]
        name, total, free, used = parts
        
        print(f"GPU {i}: {name}")
        print(f"  Total:  {int(total):,} MB")
        print(f"  Used:   {int(used):,} MB")
        print(f"  Free:   {int(free):,} MB")
        print()
        
        # Analysis
        free_mb = int(free)
        
        # Each hybrid_v2 training needs ~7-8GB
        # For 3 parallel runs: need ~24GB free
        single_need = 8000  # 8GB per run
        parallel_need = 3 * single_need  # 24GB for 3 runs
        
        print("Training Resource Requirements:")
        print(f"  Single run:     ~{single_need:,} MB (8 GB)")
        print(f"  Parallel (×3):  ~{parallel_need:,} MB (24 GB)")
        print()
        
        if free_mb >= parallel_need:
            print("✅ PARALLEL TRAINING FEASIBLE")
            print(f"   Sufficient memory for 3 parallel runs ({free_mb:,} MB free)")
            print()
            print("Recommended: ./scripts/parallel_train.sh")
            status = 0
        elif free_mb >= parallel_need * 0.8:  # 80% threshold (19.2GB)
            print("⚠️  PARALLEL TRAINING POSSIBLE BUT TIGHT")
            print(f"   Close to limit ({free_mb:,} MB free, need ~{parallel_need:,} MB)")
            print("   May work but could OOM if other processes use GPU")
            print()
            print("Options:")
            print("  1. Try parallel: ./scripts/parallel_train.sh")
            print("  2. Safe bet:    ./scripts/sequential_train.sh")
            status = 1
        else:
            print("❌ INSUFFICIENT MEMORY FOR PARALLEL TRAINING")
            print(f"   Only {free_mb:,} MB free, need ~{parallel_need:,} MB")
            print()
            print("Recommended: ./scripts/sequential_train.sh")
            print("Alternative: Reduce --max-samples or --batch-size")
            status = 2
    
    sys.exit(status)
    
except FileNotFoundError:
    print("❌ nvidia-smi not found")
    print("   Cannot detect GPU. Are you on a system with NVIDIA GPU?")
    print()
    print("If you have GPU but nvidia-smi is not in PATH:")
    print("  - Check CUDA installation")
    print("  - Ensure drivers are installed")
    print()
    print("If no GPU available:")
    print("  - Training will use CPU (very slow)")
    print("  - Not recommended for this model size")
    sys.exit(3)
    
except subprocess.CalledProcessError as e:
    print(f"❌ nvidia-smi failed: {e}")
    print()
    print("This might mean:")
    print("  - No GPU available")
    print("  - GPU drivers not properly installed")
    print("  - GPU in use by another process")
    sys.exit(3)
