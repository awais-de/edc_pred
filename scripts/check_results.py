#!/usr/bin/env python3
"""
Simple script to list all available results in experiments directory.
"""

from pathlib import Path
import json

experiments_dir = Path("experiments")

if not experiments_dir.exists():
    print("experiments/ directory not found")
    exit(1)

print("\n" + "="*100)
print("AVAILABLE RESULTS IN EXPERIMENTS DIRECTORY")
print("="*100 + "\n")

run_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])

if not run_dirs:
    print("No experiment directories found")
    exit(1)

print(f"Total runs: {len(run_dirs)}\n")

for run_dir in run_dirs:
    print(f"📁 {run_dir.name}/")
    
    # Check what files exist
    files_present = []
    
    if (run_dir / "metadata.json").exists():
        files_present.append("✓ metadata.json")
    
    if (run_dir / "predictions.npy").exists():
        files_present.append("✓ predictions.npy")
    
    if (run_dir / "targets.npy").exists():
        files_present.append("✓ targets.npy")
    
    if (run_dir / "scaler_X.pkl").exists():
        files_present.append("✓ scaler_X.pkl")
    
    if (run_dir / "scaler_y.pkl").exists():
        files_present.append("✓ scaler_y.pkl")
    
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            files_present.append(f"✓ checkpoints/ ({len(ckpt_files)} file(s))")
    
    for file_status in files_present:
        print(f"   {file_status}")
    
    print()

print("="*100)
