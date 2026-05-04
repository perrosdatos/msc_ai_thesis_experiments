#!/bin/bash
# ==============================================================================
# Full Thesis Cross-Play Sweep Execution Script
# ==============================================================================
# This script automates the complete benchmarking pipeline for the thesis.
# It performs the following steps:
# 1. Cleans the environment of any previous data and HTML reports.
# 2. Runs the cross-play evaluation across ALL 40 checkpoints using the 50 seeds.
# 3. Generates the interactive visualization dashboards.
# ==============================================================================

# Fail on error
set -e

# Change to the experiments directory
cd /home/carlos/Documents/github/msc_ai_thesis_experiments

#echo "======================================================"
#echo "🧹 Phase 1: Cleaning previous sweep data"
#echo "======================================================"
# python performance_analysis/0_clean_sweep_data.py

echo ""
echo "======================================================"
echo "⚔️ Phase 2: Running Full Cross-Play Sweep"
echo "   Evaluating ALL 40 checkpoints across 50 seeds."
echo "   This will take a significant amount of time."
echo "======================================================"

# Generate a comma-separated list of checkpoints from 1 to 40: "1,2,3,...,40"
CHECKPOINTS=$(seq -s, 1 40)

# Run the sweep
PYTHONPATH=. python performance_analysis/1_run_cross_play_sweep.py --checkpoints $CHECKPOINTS

echo ""
echo "======================================================"
echo "📊 Phase 3: Generating Visualizations & Dashboards"
echo "======================================================"
PYTHONPATH=. python performance_analysis/2_generate_visualizations.py

echo ""
echo "✅ FULL PIPELINE COMPLETE."
echo "Dashboards are available at: performance_analysis/html_reports/sweep_dashboards"
