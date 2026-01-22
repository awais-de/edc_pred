#!/usr/bin/env python3
"""Quick diagnostic to list experiments directory."""

from pathlib import Path
import json

exp_path = Path("experiments")

if not exp_path.exists():
    print("experiments/ not found")
    exit(1)

dirs_with_metadata = []
checked = 0

for item in sorted(exp_path.iterdir()):
    if item.is_dir():
        checked += 1
        metadata_file = item / "metadata.json"
        if metadata_file.exists():
            dirs_with_metadata.append(item.name)

print(f"Checked: {checked} directories")
print(f"With metadata.json: {len(dirs_with_metadata)}")
print(f"\nRun directories:")
for run_id in dirs_with_metadata:
    print(f"  - {run_id}")
