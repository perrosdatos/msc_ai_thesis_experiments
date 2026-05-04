#!/bin/bash

# Navigate to the root directory of the repository
cd "$(dirname "$0")/.."

echo "=========================================================="
echo "🤖 Starting Rule-Based Evaluation Pipeline..."
echo "=========================================================="

# Activate the virtual environment
source .venv/bin/activate

# Set PYTHONPATH so Python can find the performance_analysis module
export PYTHONPATH=.

# Step 1: Run the evaluation script
echo ""
echo "[1/2] ⚔️  Evaluating Top 5 checkpoints of each model against the Rule-Based agent..."
python performance_analysis/3_evaluate_rulebased.py

# Step 2: Generate the visualizations
echo ""
echo "[2/2] 📊 Generating Rule-Based visualization dashboards..."
python performance_analysis/4_generate_rulebased_visualizations.py

echo ""
echo "✅ Rule-Based Pipeline Complete!"
echo "📍 Dashboards are available in: performance_analysis/html_reports/rulebased_dashboards/"
