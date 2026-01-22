#!/usr/bin/env python3
"""
Monitor parallel training progress by checking the latest lines of log files.
"""

import os
import re
from pathlib import Path
import time

LOG_FILES = [
    'run1_conservative.log',
    'run2_moderate.log', 
    'run3_aggressive.log'
]

def tail_file(filepath, n=3):
    """Read last n lines of a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except FileNotFoundError:
        return []

def extract_epoch_info(lines):
    """Extract current epoch and metrics from log lines."""
    for line in reversed(lines):
        # Look for "Epoch XX:" pattern
        match = re.search(r'Epoch\s+(\d+):', line)
        if match:
            epoch = int(match.group(1))
            
            # Try to extract val_loss
            val_loss_match = re.search(r'val_loss=([\d.]+)', line)
            val_loss = float(val_loss_match.group(1)) if val_loss_match else None
            
            # Try to extract val_mae
            val_mae_match = re.search(r'val_mae=([\d.]+)', line)
            val_mae = float(val_mae_match.group(1)) if val_mae_match else None
            
            return epoch, val_loss, val_mae
    
    return None, None, None

def get_run_status(log_file):
    """Get current status of a training run."""
    if not os.path.exists(log_file):
        return "Not started", None, None, None
    
    lines = tail_file(log_file, n=10)
    
    if not lines:
        return "Empty log", None, None, None
    
    # Check if completed
    for line in lines:
        if "TRAINING COMPLETE" in line or "Training completed in" in line:
            epoch, val_loss, val_mae = extract_epoch_info(lines)
            return "✅ Complete", epoch, val_loss, val_mae
    
    # Check for errors
    for line in lines:
        if "Error" in line or "RuntimeError" in line or "CUDA out of memory" in line:
            return "❌ Failed", None, None, None
    
    # Otherwise, extract current progress
    epoch, val_loss, val_mae = extract_epoch_info(lines)
    if epoch is not None:
        return f"🔄 Epoch {epoch}/200", epoch, val_loss, val_mae
    
    return "🔄 Running", None, None, None

def main():
    """Monitor all parallel training runs."""
    print("=" * 80)
    print("PARALLEL TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print()
    
    # Configuration mapping
    configs = {
        'run1_conservative.log': 'Conservative (EDT=1.5, T20=2.5, C50=2.0)',
        'run2_moderate.log': 'Moderate (EDT=1.5, T20=3.0, C50=2.5)',
        'run3_aggressive.log': 'Aggressive (EDT=1.5, T20=4.0, C50=3.5)'
    }
    
    while True:
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 80)
        print(f"PARALLEL TRAINING PROGRESS - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        all_complete = True
        any_running = False
        any_failed = False
        
        for log_file in LOG_FILES:
            config_name = configs.get(log_file, log_file)
            status, epoch, val_loss, val_mae = get_run_status(log_file)
            
            print(f"Run: {config_name}")
            print(f"  Status: {status}")
            
            if epoch is not None:
                print(f"  Progress: {epoch}/200 epochs ({epoch/200*100:.1f}%)")
                if val_loss is not None:
                    print(f"  Val Loss: {val_loss:.3f}")
                if val_mae is not None:
                    print(f"  Val MAE: {val_mae:.4f}")
            
            if "Complete" not in status:
                all_complete = False
            if "Running" in status or "Epoch" in status:
                any_running = True
            if "Failed" in status:
                any_failed = True
            
            print()
        
        print("-" * 80)
        
        if all_complete:
            print("🎉 All runs completed!")
            print()
            print("Next steps:")
            print("  python scripts/compare_runs.py --sort-by mae --top 10")
            break
        elif any_failed:
            print("⚠️  At least one run has failed. Check logs for details.")
            print()
            print("To view logs:")
            print("  cat run1_conservative.log")
            print("  cat run2_moderate.log")
            print("  cat run3_aggressive.log")
            if any_running:
                print()
                print("Other runs still running. Press Ctrl+C to stop monitoring.")
                time.sleep(30)
            else:
                break
        elif any_running:
            print("⏳ Training in progress... Refreshing every 30 seconds.")
            print("   Press Ctrl+C to stop monitoring (training will continue).")
            time.sleep(30)
        else:
            print("⏸️  No runs detected. Waiting...")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("Monitoring stopped. Training continues in background.")
        print()
        print("To resume monitoring: python scripts/monitor_training.py")
        print("To check logs manually: tail -f run*.log")
