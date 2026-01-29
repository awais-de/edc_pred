#!/bin/bash
#
# EDC Prediction - Quick Start Script
# Usage: bash quickstart.sh [command]
#
# Commands:
#   predict-sample   - Make prediction on sample room
#   evaluate         - Run full evaluation
#   all              - Run all checks
#   help             - Show this message

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}EDC PREDICTION - QUICK START${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if dependencies are installed
check_dependencies() {
    echo -e "\n${BLUE}Checking dependencies...${NC}"
    
    if ! python -c "import torch" 2>/dev/null; then
        print_warn "PyTorch not found. Installing..."
        pip install -r requirements.txt
    else
        print_status "PyTorch found"
    fi
    
    if ! python -c "import pytorch_lightning" 2>/dev/null; then
        print_warn "PyTorch Lightning not found. Installing..."
        pip install -r requirements.txt
    else
        print_status "PyTorch Lightning found"
    fi
}

# Function to run sample prediction
predict_sample() {
    echo -e "\n${BLUE}Running sample prediction...${NC}"
    
    CHECKPOINT="experiments/multihead_20260123_120009/checkpoints/best_model.ckpt"
    FEATURES="data/raw/roomFeaturesDataset.csv"
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_warn "Checkpoint not found at $CHECKPOINT"
        echo "Available experiments:"
        ls -la experiments/ | grep "^d" | awk '{print $NF}'
        return 1
    fi
    
    if [ ! -f "$FEATURES" ]; then
        print_warn "Features file not found at $FEATURES"
        return 1
    fi
    
    print_status "Checkpoint: $CHECKPOINT"
    print_status "Features: $FEATURES"
    
    echo ""
    python inference.py \
        --checkpoint "$CHECKPOINT" \
        --features "$FEATURES" \
        --index 0
    
    print_status "Sample prediction completed!"
}

# Function to run full evaluation
run_evaluation() {
    echo -e "\n${BLUE}Running full evaluation...${NC}"
    
    CHECKPOINT="experiments/multihead_20260123_120009/checkpoints/best_model.ckpt"
    FEATURES="data/raw/roomFeaturesDataset.csv"
    EDC_DIR="data/raw/EDC"
    OUTPUT_DIR="results"
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_warn "Checkpoint not found"
        return 1
    fi
    
    print_info "This may take 5-10 minutes depending on your hardware..."
    echo ""
    
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --features "$FEATURES" \
        --edc-dir "$EDC_DIR" \
        --output "$OUTPUT_DIR"
    
    echo ""
    print_status "Evaluation completed!"
    print_info "Results saved to: $OUTPUT_DIR/"
    
    if [ -f "$OUTPUT_DIR/metrics_table.csv" ]; then
        echo ""
        echo -e "${BLUE}Metrics Summary:${NC}"
        cat "$OUTPUT_DIR/metrics_table.csv"
    fi
}

# Function to show help
show_help() {
    echo -e "${BLUE}EDC Prediction - Quick Start Commands${NC}\n"
    
    echo "Available commands:"
    echo ""
    echo "  $(tput bold)predict-sample$(tput sgr0)  - Make prediction on a single room"
    echo "  $(tput bold)evaluate$(tput sgr0)        - Run full evaluation and generate plots"
    echo "  $(tput bold)all$(tput sgr0)             - Check dependencies and run all tests"
    echo "  $(tput bold)help$(tput sgr0)            - Show this help message"
    echo ""
    
    echo "Usage:"
    echo "  bash quickstart.sh predict-sample"
    echo "  bash quickstart.sh evaluate"
    echo "  bash quickstart.sh all"
    echo ""
    
    echo "Or run Python directly:"
    echo "  python inference.py --index 0"
    echo "  python evaluate.py"
    echo ""
}

# Main logic
cd "$PROJECT_ROOT"

COMMAND="${1:-help}"

case "$COMMAND" in
    predict-sample)
        check_dependencies
        predict_sample
        ;;
    evaluate)
        check_dependencies
        run_evaluation
        ;;
    all)
        check_dependencies
        predict_sample
        echo ""
        run_evaluation
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ Done!${NC}"
echo -e "${GREEN}================================================${NC}\n"
