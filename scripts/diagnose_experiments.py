"""
Diagnostic script to show the experiments directory structure.
"""

from pathlib import Path
import os

experiments_path = Path("experiments")

if not experiments_path.exists():
    print("Experiments directory not found")
    exit(1)

print("Experiments directory contents:\n")

# List all top-level items
top_level = sorted(experiments_path.iterdir())
print(f"Total items in experiments/: {len(top_level)}\n")

for item in top_level:
    if item.is_dir():
        # Count files in this directory
        files = list(item.iterdir())
        metadata_exists = (item / "metadata.json").exists()
        print(f"📁 {item.name}/")
        print(f"   Items: {len(files)}")
        print(f"   Has metadata.json: {metadata_exists}")
        
        # List first few items
        for subitem in sorted(files)[:5]:
            if subitem.is_file():
                print(f"     📄 {subitem.name}")
            else:
                print(f"     📁 {subitem.name}/")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more items")
    else:
        print(f"📄 {item.name}")
    print()
