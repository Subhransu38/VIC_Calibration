#!/bin/bash

# Enhanced VIC Calibration Runner Script
# Usage: ./run_calibration.sh [iterations] [algorithm] [processes]

set -e  # Exit on any error

# Default values
ITERATIONS=${1:-500}
ALGORITHM=${2:-sceua}
PROCESSES=${3:-1}
CONFIG_FILE="config/calibration_config.ini"
LOG_LEVEL="INFO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    VIC Multi-Station Calibration Runner${NC}"
echo -e "${BLUE}================================================${NC}"
echo
echo -e "Configuration:"
echo -e "  Iterations: ${GREEN}${ITERATIONS}${NC}"
echo -e "  Algorithm:  ${GREEN}${ALGORITHM}${NC}"
echo -e "  Processes:  ${GREEN}${PROCESSES}${NC}"
echo -e "  Config:     ${GREEN}${CONFIG_FILE}${NC}"
echo

# Check if required files exist
echo -e "${YELLOW}Checking prerequisites...${NC}"

required_files=(
    "calibration_enhanced.py"
    "format_soil_params_final.py"
    "$CONFIG_FILE"
    "routing_model/rout"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}Error: Required file not found: $file${NC}"
        exit 1
    fi
done

# Check if required directories exist
required_dirs=(
    "input"
    "output"
    "src"
    "routing_model"
)

for dir in "${required_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo -e "${YELLOW}Creating directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Make executables executable
chmod +x routing_model/rout 2>/dev/null || true
chmod +x input/vicNl 2>/dev/null || true

echo -e "${GREEN}Prerequisites check completed.${NC}"
echo

# Backup previous results if they exist
BACKUP_DIR="backup/results/$(date +%Y%m%d_%H%M%S)"
if [[ -d "output/calibration/results" ]] && [[ -n "$(ls -A output/calibration/results 2>/dev/null)" ]]; then
    echo -e "${YELLOW}Backing up previous results to $BACKUP_DIR${NC}"
    mkdir -p "$BACKUP_DIR"
    cp -r output/calibration/results/* "$BACKUP_DIR/" 2>/dev/null || true
fi

# Run calibration
echo -e "${BLUE}Starting calibration...${NC}"
echo

# Activate virtual environment if it exists
if [[ -f "venv/bin/activate" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the calibration
python calibration_enhanced.py \
    --iterations "$ITERATIONS" \
    --algorithm "$ALGORITHM" \
    --processes "$PROCESSES" \
    --config "$CONFIG_FILE" \
    --log-level "$LOG_LEVEL"

calibration_exit_code=$?

if [[ $calibration_exit_code -eq 0 ]]; then
    echo
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}    Calibration completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo
    echo -e "Results saved in: ${GREEN}output/calibration/results/${NC}"
    echo -e "Logs saved in:    ${GREEN}output/calibration/logs/${NC}"
    echo
    
    # Generate summary report if script exists
    if [[ -f "scripts/analysis/generate_summary.py" ]]; then
        echo -e "${YELLOW}Generating summary report...${NC}"
        python scripts/analysis/generate_summary.py --config "$CONFIG_FILE"
    fi
    
else
    echo
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}    Calibration failed!${NC}"
    echo -e "${RED}================================================${NC}"
    echo
    echo -e "Check logs in: ${RED}output/calibration/logs/${NC}"
    echo -e "Exit code: ${RED}$calibration_exit_code${NC}"
    exit $calibration_exit_code
fi