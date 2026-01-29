#!/bin/bash

################################################################################
# PROJECT CLEANUP SCRIPT: Move non-whitelisted files to backup directory
################################################################################
#
# PURPOSE:
#   Removes all files and directories from a Git project EXCEPT those on the
#   whitelist, moving them to a backup directory one level above the project.
#
# BEHAVIOR:
#   - Project: /path/to/project
#   - Backup:  /path/to/project_backup
#
#   After running, only whitelisted files remain in the project directory.
#   Everything else is preserved in the backup directory with the same
#   relative path structure.
#
# USAGE:
#   1. Set PROJECT_ROOT (below) to your project directory path
#   2. Update WHITELIST (below) with files/directories to keep
#   3. Run: bash cleanup_project.sh
#
# SAFETY FEATURES:
#   - Prints all moves before executing
#   - Asks for confirmation before proceeding
#   - Fails fast on errors (set -e)
#   - Checks for existing backup directory
#   - Preserves relative directory structure
#
################################################################################

set -e

################################################################################
# CONFIGURATION - EDIT THESE
################################################################################

# ============================================================================
# PROJECT_ROOT: Full path to the project directory
# ============================================================================
# For local use:
# PROJECT_ROOT="/Users/muhammadawais/Downloads/ADSP/proj/edc_pred"
#
# For SSH/remote use, set this to the absolute path on the remote server:
PROJECT_ROOT="${1:-.}"

# ============================================================================
# WHITELIST: Array of relative paths to KEEP in the project
# ============================================================================
# These paths will NOT be moved to backup. Everything else will be moved.
# Paths are relative to PROJECT_ROOT.
#
# Example:
#   WHITELIST=(
#       "requirements.txt"
#       "train_multihead.py"
#       "src/models/multihead_model.py"
#       "src/models/__init__.py"
#       "data/raw/roomFeaturesDataset.csv"
#   )
#
read -r -d '' WHITELIST << 'EOF' || true
requirements.txt
train_multihead.py
src
data
.git
.gitignore
.venv
venv
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
CLEANUP_SCRIPT_DOCUMENTATION.md
scripts
cleanup_project.sh
EOF

# ============================================================================
# BACKUP_SUFFIX: Name of backup directory (relative to parent of PROJECT_ROOT)
# ============================================================================
# If PROJECT_ROOT is /path/to/project, backup will be created at:
#   /path/to/${PROJECT_ROOT##*/}${BACKUP_SUFFIX}
# Default: "_backup"
BACKUP_SUFFIX="_backup"

# ============================================================================
# DRY_RUN: Set to true to preview changes without moving files
# ============================================================================
DRY_RUN=false

# ============================================================================
# SKIP_CONFIRMATION: Set to true to skip the confirmation prompt
# ============================================================================
# WARNING: Only use this if you know what you're doing!
SKIP_CONFIRMATION=false

################################################################################
# SCRIPT STARTS HERE
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# VALIDATE INPUTS
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PROJECT CLEANUP SCRIPT${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if PROJECT_ROOT exists
if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}❌ ERROR: Project root does not exist:${NC}"
    echo "   $PROJECT_ROOT"
    exit 1
fi

# Resolve to absolute path
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
PROJECT_NAME="${PROJECT_ROOT##*/}"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
BACKUP_DIR="${PARENT_DIR}/${PROJECT_NAME}${BACKUP_SUFFIX}"

echo -e "${GREEN}✓ Project root:${NC} $PROJECT_ROOT"
echo -e "${GREEN}✓ Project name:${NC} $PROJECT_NAME"
echo -e "${GREEN}✓ Parent dir:${NC} $PARENT_DIR"
echo -e "${GREEN}✓ Backup dir:${NC} $BACKUP_DIR"
echo ""

# ============================================================================
# CONVERT WHITELIST TO ARRAY
# ============================================================================

# Convert multiline whitelist to array (compatible with all bash versions)
WHITELIST_ARRAY=()
while IFS= read -r line; do
    [ -z "$line" ] && continue
    WHITELIST_ARRAY+=("$line")
done <<< "$WHITELIST"

echo -e "${BLUE}Whitelisted paths (will remain):${NC}"
for item in "${WHITELIST_ARRAY[@]}"; do
    [ -z "$item" ] && continue
    echo "  • $item"
done
echo ""

# ============================================================================
# FIND FILES TO MOVE
# ============================================================================

echo -e "${BLUE}Scanning for files to move...${NC}"
echo ""

# Build a list of files to move (top-level items only)
FILES_TO_MOVE=()

# Walk through project directory (exclude .venv from find to speed up scanning)
# Only process mindepth 1 and maxdepth 1 to get top-level items only
while IFS= read -r -d '' file; do
    # Get relative path
    rel_path="${file#$PROJECT_ROOT/}"
    
    # Skip if it's the project root itself
    [ "$rel_path" = "" ] && continue
    
    # Check if this file/dir is in the whitelist
    is_whitelisted=false
    for wl_item in "${WHITELIST_ARRAY[@]}"; do
        [ -z "$wl_item" ] && continue
        
        # Exact match
        if [ "$rel_path" = "$wl_item" ]; then
            is_whitelisted=true
            break
        fi
        
        # Parent directory match (e.g., src/* matches src)
        if [ "${rel_path:0:${#wl_item}}" = "$wl_item" ] && \
           ([ "${rel_path:${#wl_item}:1}" = "/" ] || [ ${#rel_path} -eq ${#wl_item} ]); then
            is_whitelisted=true
            break
        fi
    done
    
    # If not whitelisted, add to move list
    if ! $is_whitelisted; then
        FILES_TO_MOVE+=("$rel_path")
    fi
done < <(find "$PROJECT_ROOT" -mindepth 1 -maxdepth 1 ! -name ".venv" ! -name ".git" -print0)

# Note: Only top-level items are listed to avoid moving nested items twice

if [ ${#FILES_TO_MOVE[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠ No files need to be moved.${NC}"
    echo ""
    exit 0
fi

echo -e "${YELLOW}Files and directories to MOVE:${NC}"
echo ""
for file in "${FILES_TO_MOVE[@]}"; do
    if [ -d "$PROJECT_ROOT/$file" ]; then
        echo -e "  ${YELLOW}📁${NC} $file/"
    else
        echo -e "  ${YELLOW}📄${NC} $file"
    fi
done
echo ""

# ============================================================================
# CONFIRM AFTER LISTING
# ============================================================================

read -p "Review the list above. Continue? (y/n): " -r confirm_list
echo ""

if [ "$confirm_list" != "y" ] && [ "$confirm_list" != "yes" ]; then
    echo -e "${RED}Cancelled after file review.${NC}"
    exit 0
fi

# ============================================================================
# SHOW SUMMARY
# ============================================================================

echo -e "${BLUE}Summary:${NC}"
echo "  Total files/dirs to move: ${#FILES_TO_MOVE[@]}"
echo "  Backup destination: $BACKUP_DIR"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}⚠ DRY RUN MODE: No files will actually be moved.${NC}"
    echo ""
fi

# ============================================================================
# ASK FOR CONFIRMATION
# ============================================================================

if ! $SKIP_CONFIRMATION; then
    echo -e "${YELLOW}⚠ This operation will move files out of Git control.${NC}"
    echo ""
    read -p "Continue? (yes/no): " -r confirm
    echo ""
    
    if [ "$confirm" != "yes" ]; then
        echo -e "${RED}Cancelled.${NC}"
        exit 0
    fi
fi

# ============================================================================
# CHECK FOR EXISTING BACKUP
# ============================================================================

if [ -d "$BACKUP_DIR" ]; then
    echo -e "${YELLOW}⚠ Backup directory already exists:${NC}"
    echo "   $BACKUP_DIR"
    echo ""
    read -p "Overwrite? (yes/no): " -r overwrite
    echo ""
    
    if [ "$overwrite" != "yes" ]; then
        echo -e "${RED}Cancelled.${NC}"
        exit 0
    fi
else
    echo -e "${GREEN}Creating backup directory...${NC}"
    if ! $DRY_RUN; then
        mkdir -p "$BACKUP_DIR"
    fi
    echo ""
fi

# ============================================================================
# MOVE FILES
# ============================================================================

echo -e "${BLUE}Moving files...${NC}"
echo ""

moved_count=0
for file in "${FILES_TO_MOVE[@]}"; do
    src="$PROJECT_ROOT/$file"
    dest="$BACKUP_DIR/$file"
    
    if [ -d "$src" ]; then
        echo -e "  ${YELLOW}📁 Moving directory:${NC} $file"
        if ! $DRY_RUN; then
            mkdir -p "$(dirname "$dest")"
            mv "$src" "$dest"
        fi
    else
        echo -e "  ${YELLOW}📄 Moving file:${NC} $file"
        if ! $DRY_RUN; then
            mkdir -p "$(dirname "$dest")"
            mv "$src" "$dest"
        fi
    fi
    
    ((moved_count++))
done

echo ""

# ============================================================================
# VERIFY CLEANUP
# ============================================================================

if ! $DRY_RUN; then
    echo -e "${BLUE}Verifying cleanup...${NC}"
    echo ""
    
    # Count remaining files in project (excluding .git)
    remaining=$(find "$PROJECT_ROOT" -mindepth 1 ! -path "*/.git*" -type f 2>/dev/null | wc -l)
    
    echo -e "${GREEN}✓ Cleanup complete!${NC}"
    echo "  Moved: $moved_count items"
    echo "  Remaining in project: $remaining files"
    echo ""
    
    echo -e "${GREEN}Project structure now contains only:${NC}"
    find "$PROJECT_ROOT" -mindepth 1 ! -path "*/.git*" -type f -o -type d 2>/dev/null | \
        sed "s|^$PROJECT_ROOT/|  |" | sort
    echo ""
    
    echo -e "${GREEN}Backup structure:${NC}"
    find "$BACKUP_DIR" -mindepth 1 -type f -o -type d 2>/dev/null | \
        sed "s|^$BACKUP_DIR/|  |" | sort
    echo ""
else
    echo -e "${YELLOW}DRY RUN COMPLETE: No changes were made.${NC}"
    echo ""
fi

# ============================================================================
# FINAL STATUS
# ============================================================================

echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Cleanup script completed successfully!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# CLEANUP INSTRUCTIONS FOR GIT
# ============================================================================

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Verify the whitelisted files are still in the project"
echo "  2. Check git status:"
echo "     cd $PROJECT_ROOT"
echo "     git status"
echo "  3. If files were deleted, commit the cleanup:"
echo "     git add -A"
echo "     git commit -m 'Cleanup: Move non-essential files to backup'"
echo ""

exit 0
