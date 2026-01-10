#!/usr/bin/env python3
"""
Create standard folder structure for
'generalized_edc_prediction' project.
Works for both CNN-LSTM and Transformer variants.
"""

import os
from pathlib import Path

PROJECT_NAME = "edc_pred"

# Root of the repo (run this from the repo root)
ROOT = Path(".").resolve()

# Folders to create (relative to ROOT)
DIRS = [
    # Data
    "data/raw",          # Original EDC .npy + roomFeatures CSV
    "data/processed",    # Scaled / split datasets
    "data/external",     # Measured data, extra corpora

    # Experiment artifacts
    "experiments/checkpoints",   # Saved model weights
    "experiments/logs",          # TensorBoard / Lightning logs
    "experiments/figures",       # Plots: EDCs, losses, metrics
    "experiments/configs",       # YAML/JSON configs for runs

    # Source code package
    f"src/{PROJECT_NAME}",
    f"src/{PROJECT_NAME}/data",        # Loading, preprocessing, splitting
    f"src/{PROJECT_NAME}/models",      # CNN-LSTM, Transformer, losses
    f"src/{PROJECT_NAME}/training",    # Trainer, Lightning modules, loops
    f"src/{PROJECT_NAME}/evaluation",  # Metric computation (EDT, T20, C50)
    f"src/{PROJECT_NAME}/visualization",  # Plotting utilities
    f"src/{PROJECT_NAME}/utils",       # Common helpers, config, logging

    # Notebooks & scripts
    "notebooks",               # EDA, prototyping
    "scripts",                 # CLI scripts: train.py, evaluate.py
]

# Files to create (relative path, with optional initial content)
FILES = {
    f"src/{PROJECT_NAME}/__init__.py": "",
    f"src/{PROJECT_NAME}/data/__init__.py": "",
    f"src/{PROJECT_NAME}/models/__init__.py": "",
    f"src/{PROJECT_NAME}/training/__init__.py": "",
    f"src/{PROJECT_NAME}/evaluation/__init__.py": "",
    f"src/{PROJECT_NAME}/visualization/__init__.py": "",
    f"src/{PROJECT_NAME}/utils/__init__.py": "",

    # Entry points / templates
    "scripts/train_edc.py": "# Entry script to train CNN-LSTM or Transformer model\n",
    "scripts/evaluate_edc.py": "# Script to evaluate models and compute EDT/T20/C50\n",
    "requirements.txt": "# Add Python package requirements here (pytorch, pytorch-lightning, numpy, etc.)\n",
    "README.md": f"# {PROJECT_NAME}\n\nProject for generalized prediction of Energy Decay Curves (EDCs) from room geometry using deep neural networks.\n",
    ".gitignore": "data/raw/\ndata/processed/\nexperiments/checkpoints/\nexperiments/logs/\n",
}

def main():
    # Create directories
    for d in DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)

    # Create files
    for rel_path, content in FILES.items():
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    print(f"Project structure created for '{PROJECT_NAME}' at {ROOT}")

if __name__ == "__main__":
    main()
