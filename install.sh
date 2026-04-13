#!/bin/bash

# Exit on error
set -e

echo "==========================================="
echo "   Lux AI S3 PPO Agent Setup (Python 3.11) "
echo "==========================================="

PY311_BIN="/home/carlos/.pyenv/versions/3.11.14/bin/python"

# Ensure Python 3.11.14 is available
echo "[0/5] Verifying Python 3.11.14..."
if [ -f "$PY311_BIN" ]; then
    echo "Using Python 3.11.14 from pyenv: $PY311_BIN"
else
    echo "Error: Python 3.11.14 not found at $PY311_BIN. Please ensure it was installed correctly."
    exit 1
fi

echo "[1/5] Creating fresh virtual environment '.venv'..."
rm -rf .venv
$PY311_BIN -m venv .venv

echo "[2/5] Upgrading core build tools..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "[3/5] Installing project dependencies..."
# Install from requirements.txt
pip install -r requirements.txt

echo "[4/5] Installing CUDA-accelerated JAX..."
# Install JAX with CUDA 12 support (Standard for RTX 3080 Ti)
pip install -U "jax[cuda12]"

echo "[5/5] Finalizing setup..."

echo "==========================================="
echo "Setup Complete!"
echo "To verify GPU acceleration, run:"
echo "    source .venv/bin/activate"
echo "    python -c 'import jax; print(jax.devices())'"
echo "==========================================="
echo "To begin training:"
echo "    python train_ppo_lux.py"
