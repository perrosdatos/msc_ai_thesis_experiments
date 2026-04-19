#!/bin/bash
# Wrapper to generate all reports
set -e

# Load python environment
source .venv/bin/activate

# Execute documentation generator
export GIT_HASH=$(git rev-parse --short HEAD || echo "unknown")
export STAMP=$(date +"%Y%m%d_%H%M%S")
export REPORT_OUT_DIR="/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/${GIT_HASH}_${STAMP}"

mkdir -p "$REPORT_OUT_DIR"

python generate_documentation.py
python generate_architecture_report.py