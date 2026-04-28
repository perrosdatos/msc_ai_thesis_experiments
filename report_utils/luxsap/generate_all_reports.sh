#!/bin/bash
# Wrapper to generate all reports for LUXSAP Combat Environment
set -e

# Change to the script's directory automatically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load python environment (2 levels up from script)
source ../../.venv/bin/activate

# Execute documentation generator
export GIT_HASH=$(git rev-parse --short HEAD || echo "unknown")
export STAMP=$(date +"%Y%m%d_%H%M%S")
export REPORT_OUT_DIR="../../html_reports/luxsap_${GIT_HASH}_${STAMP}"

mkdir -p "$REPORT_OUT_DIR"

python generate_documentation.py
python generate_architecture_report.py
python generate_reward_landscape.py
