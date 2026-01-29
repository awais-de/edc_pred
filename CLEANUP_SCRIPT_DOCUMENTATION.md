# 🧹 PROJECT CLEANUP SCRIPT - DOCUMENTATION

**Script:** [cleanup_project.sh](cleanup_project.sh)  
**Purpose:** Remove non-essential files from Git project, backup everything else  
**Status:** ✅ Ready to use

---

## 📋 QUICK START

### **Simplest Usage (Local Project)**

```bash
bash cleanup_project.sh
```

This will:
1. Show you what files will be moved
2. Ask for confirmation
3. Move everything NOT on the whitelist to `../project_backup`

### **Remote/SSH Usage**

```bash
# Run on remote server (via SSH)
ssh user@remote.server "cd /path/to/project && bash cleanup_project.sh"

# Or copy script first, then run
scp cleanup_project.sh user@remote.server:/path/to/project/
ssh user@remote.server "cd /path/to/project && bash ./cleanup_project.sh"
```

---

## 🎯 WHAT THIS SCRIPT DOES

### **Before Running**
```
project/
  ├── requirements.txt          ✅ Keep
  ├── train_multihead.py        ✅ Keep
  ├── cleanup_project.sh        ✅ Keep
  ├── .git/                     ✅ Keep
  ├── src/                      ✅ Keep (entire directory)
  ├── data/raw/roomFeaturesDataset.csv  ✅ Keep
  ├── CONVERSATION_CONTEXT.md   ✅ Keep
  ├── RESULTS_ANALYSIS.md       ✅ Keep
  ├── .DS_Store                 ❌ Move to backup
  ├── models/old/               ❌ Move to backup
  ├── notebooks/                ❌ Move to backup
  ├── inference.py              ❌ Move to backup
  └── ... (30+ other files)     ❌ Move to backup
```

### **After Running**
```
project/
  ├── requirements.txt          ✅
  ├── train_multihead.py        ✅
  ├── cleanup_project.sh        ✅
  ├── .git/                     ✅
  ├── src/                      ✅
  ├── data/raw/
  │   └── roomFeaturesDataset.csv  ✅
  └── [20+ documentation files] ✅

../project_backup/
  ├── .DS_Store
  ├── models/old/
  ├── notebooks/
  ├── inference.py
  ├── test_allowed_architectures.py
  ├── validate_architectures.py
  ├── train_model.py
  ├── lstm_paper.pdf
  ├── STATUS.txt
  └── ... (all other non-whitelisted items)
```

**Result:** Git project is clean, Git history is not affected, backup preserved.

---

## ⚙️ CONFIGURATION

### **Setting the Project Root**

The script defaults to current directory (`.`), but you can specify:

```bash
# Option 1: Argument (recommended)
bash cleanup_project.sh /path/to/project

# Option 2: Edit the script directly
# Change this line in cleanup_project.sh:
PROJECT_ROOT="/Users/muhammadawais/Downloads/ADSP/proj/edc_pred"
```

### **Modifying the Whitelist**

Edit the `WHITELIST` section in `cleanup_project.sh`:

```bash
read -r -d '' WHITELIST << 'EOF' || true
requirements.txt
train_multihead.py
src
data/raw/roomFeaturesDataset.csv
.git
.gitignore
README.md
# Add more lines here...
EOF
```

**Current whitelist includes:**
- Core training script: `train_multihead.py`
- All source code: `src/` (entire directory)
- Dataset: `data/raw/roomFeaturesDataset.csv`
- Documentation: All `.md` files needed for submission
- Git: `.git/` directory
- Utilities: `scripts/plot_results.py`, `scripts/compare_runs.py`
- This cleanup script: `cleanup_project.sh`

### **Customizing Backup Directory Name**

```bash
# In cleanup_project.sh, change:
BACKUP_SUFFIX="_backup"

# To something else:
BACKUP_SUFFIX="_old"
BACKUP_SUFFIX="_archived"
```

---

## 🚀 USAGE EXAMPLES

### **Example 1: Local Project (Simplest)**

```bash
cd /Users/muhammadawais/Downloads/ADSP/proj/edc_pred
bash cleanup_project.sh

# Output shows what will move, asks for confirmation
```

### **Example 2: Dry Run (Preview Without Moving)**

```bash
# Edit cleanup_project.sh, set:
DRY_RUN=true

# Then run - shows what WOULD be moved, but doesn't move
bash cleanup_project.sh

# Change back to false when ready to actually run
DRY_RUN=false
```

### **Example 3: Skip Confirmation (After Previewing)**

```bash
# Edit cleanup_project.sh, set:
SKIP_CONFIRMATION=true

# Then run - moves immediately without asking
bash cleanup_project.sh
```

### **Example 4: Remote Server via SSH**

```bash
# On local machine, copy script to remote
scp cleanup_project.sh user@remote.server:/tmp/

# SSH into remote, navigate to project, run
ssh user@remote.server
cd /path/to/project
bash /tmp/cleanup_project.sh

# Or in one command:
ssh user@remote.server "cd /path/to/project && bash cleanup_project.sh"
```

### **Example 5: Custom Project Path**

```bash
bash cleanup_project.sh /home/user/my_project

# Or on remote:
ssh user@remote.server "bash /tmp/cleanup_project.sh /home/user/my_project"
```

---

## 📊 WHAT STAYS (WHITELIST)

### **Core Code**
```
requirements.txt                           # Dependencies
train_multihead.py                         # Training script
cleanup_project.sh                         # This script
src/                                       # All source code
  ├── models/
  ├── data/
  └── evaluation/
```

### **Data**
```
data/raw/roomFeaturesDataset.csv          # Dataset
```

### **Documentation** (20+ files)
```
README.md
CONVERSATION_CONTEXT.md
RESULTS_ANALYSIS.md
COMPARATIVE_ANALYSIS.md
PROJECT_SUMMARY.md
GETTING_STARTED.md
QUICKSTART.md
FAQ_TROUBLESHOOTING.md
COMPLETION_CHECKLIST.md
SETUP_SUMMARY.md
SHIPPING_QUICK_REFERENCE.md
SHIPPING_MANIFEST.md
SHIPPING_STRUCTURE.md
SHIPPING_COMPLETE_AUDIT.md
SHIPPING_FILES_CHECKLIST.md
SHIPPING_DOCUMENTATION_INDEX.md
```

### **Utilities**
```
scripts/plot_results.py
scripts/compare_runs.py
scripts/README.md
```

### **Git**
```
.git/                                      # Git history (preserved)
.gitignore
```

---

## ❌ WHAT GETS MOVED (TO BACKUP)

### **Typical removals:**
```
.DS_Store                                  # macOS metadata
models/old/                                # Development artifacts
models/train/                              # Development artifacts
notebooks/                                 # If empty or dev-only
test_allowed_architectures.py              # Development test
validate_architectures.py                  # Development test
train_model.py                             # Old training script
inference.py                               # Stub file
lstm_paper.pdf                             # Reference paper
STATUS.txt                                 # Development notes
*.pyc                                      # Compiled Python files
__pycache__/                               # Python cache
```

---

## 🔒 SAFETY FEATURES

### **1. Fails Fast on Errors**
```bash
set -e   # Exit immediately on any error
```

### **2. Shows Everything First**
- Prints all files to be moved
- Shows source and destination
- Waits for user confirmation
- Allows review before proceeding

### **3. Dry Run Mode**
```bash
# Edit script:
DRY_RUN=true

# Shows what WOULD happen without changing anything
```

### **4. Checks Before Overwriting**
- Checks if backup dir already exists
- Asks for confirmation before overwriting
- Preserves existing backup if you say "no"

### **5. Color-Coded Output**
- 🟢 Green = Success/confirmations
- 🔴 Red = Errors
- 🟡 Yellow = Warnings/items to move
- 🔵 Blue = Headers/status

### **6. Verification After Completion**
- Shows number of items moved
- Lists remaining files in project
- Shows backup structure

---

## 🐛 TROUBLESHOOTING

### **Script won't run**

```bash
# Make sure it's executable
chmod +x cleanup_project.sh

# Then run
bash cleanup_project.sh
```

### **Permission denied errors on remote**

```bash
# SSH and check permissions
ssh user@remote.server "ls -la cleanup_project.sh"

# Make executable on remote
ssh user@remote.server "chmod +x /path/to/cleanup_project.sh"

# Then run
ssh user@remote.server "bash /path/to/cleanup_project.sh"
```

### **"Project root does not exist" error**

```bash
# Make sure path is correct
bash cleanup_project.sh /correct/path/to/project

# Or navigate first
cd /path/to/project
bash cleanup_project.sh
```

### **Want to undo the backup**

```bash
# Files are in ../project_backup/ (one level up)
# You can move them back or delete backup

# To restore everything:
cp -r ../project_backup/* .

# To just check what's in backup:
ls -la ../project_backup/
```

### **Want to modify whitelist after running**

```bash
# Edit cleanup_project.sh to change whitelist
# Don't run again (files already moved)
# 
# If you want something back:
cp -r ../project_backup/that_file .
```

---

## 📝 EXAMPLE OUTPUT

```
════════════════════════════════════════════════════════════════════
  PROJECT CLEANUP SCRIPT
════════════════════════════════════════════════════════════════════

✓ Project root: /Users/muhammadawais/Downloads/ADSP/proj/edc_pred
✓ Project name: edc_pred
✓ Parent dir: /Users/muhammadawais/Downloads/ADSP/proj
✓ Backup dir: /Users/muhammadawais/Downloads/ADSP/proj/edc_pred_backup

Whitelisted paths (will remain):
  • requirements.txt
  • train_multihead.py
  • src
  • data/raw/roomFeaturesDataset.csv
  • .git
  • ... (20+ more)

Scanning for files to move...

Files and directories to MOVE:
  📁 data/external/
  📁 data/processed/
  📁 experiments/checkpoints/
  📁 experiments/configs/
  📁 experiments/figures/
  📄 inference.py
  📁 models/old/
  📁 models/train/
  📁 notebooks/
  📄 test_allowed_architectures.py
  📄 validate_architectures.py
  📄 train_model.py
  📄 lstm_paper.pdf
  📄 STATUS.txt
  ... (15 more items)

Summary:
  Total files/dirs to move: 32
  Backup destination: /Users/muhammadawais/Downloads/ADSP/proj/edc_pred_backup

⚠ This operation will move files out of Git control.

Continue? (yes/no): yes

Creating backup directory...

Moving files...
  📁 Moving directory: data/external/
  📁 Moving directory: data/processed/
  📁 Moving directory: experiments/checkpoints/
  📄 Moving file: inference.py
  ... (28 more)

Verifying cleanup...

✓ Cleanup complete!
  Moved: 32 items
  Remaining in project: 42 files

Project structure now contains only:
  COMPARATIVE_ANALYSIS.md
  COMPLETION_CHECKLIST.md
  CONVERSATION_CONTEXT.md
  GETTING_STARTED.md
  ... (37 more)

════════════════════════════════════════════════════════════════════
✓ Cleanup script completed successfully!
════════════════════════════════════════════════════════════════════

Next steps:
  1. Verify the whitelisted files are still in the project
  2. Check git status:
     cd /Users/muhammadawais/Downloads/ADSP/proj/edc_pred
     git status
  3. If files were deleted, commit the cleanup:
     git add -A
     git commit -m 'Cleanup: Move non-essential files to backup'
```

---

## ✅ AFTER RUNNING

### **1. Verify Everything Looks Good**

```bash
# Check what's left in project
ls -la

# Check what's in backup
ls -la ../edc_pred_backup/

# Verify src still works
python -c "from src.models import get_model; print('✅ OK')"
```

### **2. Update Git**

```bash
# Check what changed
git status

# Stage the changes
git add -A

# Commit the cleanup
git commit -m "Cleanup: Move non-essential files to backup directory"

# Verify commit
git log --oneline -1
```

### **3. Optional: Push to Remote**

```bash
git push origin main
```

---

## 🎯 WORKFLOW FOR YOUR PROJECT

**Step 1: Copy script to your server**

```bash
scp cleanup_project.sh user@your.server:/path/to/edc_pred/
```

**Step 2: Run the cleanup (with preview)**

```bash
ssh user@your.server
cd /path/to/edc_pred
bash cleanup_project.sh

# Review the output, type "yes" when ready
```

**Step 3: Verify and commit**

```bash
# Still logged into remote
git status      # See what was removed
git add -A
git commit -m "Cleanup: Move development files to backup"
git log --oneline -3
```

**Step 4: Optional - backup the backup**

```bash
# If you want to keep the backup archived somewhere
tar -czf edc_pred_backup.tar.gz ../edc_pred_backup/
```

---

## 📌 IMPORTANT NOTES

1. **Git history is preserved** - Only working tree is cleaned, commits are unchanged
2. **Whitelist is customizable** - Edit the script for your needs
3. **Backup preserves structure** - Everything moved maintains relative paths
4. **Safe by default** - Asks for confirmation, shows everything first
5. **Easily reversible** - Files are just moved, not deleted
6. **Works locally or remote** - Same script works everywhere

---

## 🚀 QUICK REFERENCE

```bash
# Local usage
bash cleanup_project.sh

# Remote usage
ssh user@server "cd /path/to/project && bash cleanup_project.sh"

# Dry run (preview only)
# Edit cleanup_project.sh: DRY_RUN=true
bash cleanup_project.sh

# Skip confirmation (after previewing)
# Edit cleanup_project.sh: SKIP_CONFIRMATION=true
bash cleanup_project.sh

# Undo (if needed)
cp -r ../project_backup/* .
```

---

**Created:** January 29, 2026  
**Status:** ✅ Ready to use  
**Safety Level:** High (preview + confirmation)  
**Git Impact:** Removes files from working tree, preserves history
