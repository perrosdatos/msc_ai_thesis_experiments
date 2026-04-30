#!/bin/bash
# Script to watch the currently running MASAC process, wait 3 minutes after it finishes, and then start MAPPO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "🔍 Looking for the currently running MASAC process..."
echo "=========================================="

# Find the PID of the python process running benchmarl/run.py
MASAC_PID=$(pgrep -f "benchmarl/run.py" | head -n 1)

if [ -z "$MASAC_PID" ]; then
    echo "❌ Could not find a running MASAC process!"
    echo "Please ensure MASAC is actually running before starting this watcher."
    exit 1
fi

echo "✅ Found MASAC running with PID: $MASAC_PID"
echo "👀 Watching process $MASAC_PID. This script will wait until it finishes..."

# Wait until the process is no longer running
while kill -0 $MASAC_PID 2>/dev/null; do
    sleep 60
done

echo "=========================================="
echo "⏳ MASAC has finished! Waiting 3 minutes for system memory & VRAM to flush..."
echo "=========================================="
sleep 180

echo "=========================================="
echo "🧹 Checking for any lingering Python processes..."
echo "=========================================="
# Forcefully terminate any lingering Python processes running benchmarl to ensure VRAM is 100% free
if pgrep -f "benchmarl/run.py" > /dev/null; then
    echo "⚠️ Found lingering processes! Terminating them to free memory..."
    pkill -9 -f "benchmarl/run.py"
    sleep 5
else
    echo "✅ No lingering processes found. Memory is clean!"
fi

echo "=========================================="
echo "🚀 RUNNING MAPPO RESUME SCRIPT..."
echo "=========================================="
$SCRIPT_DIR/resume_mappo.sh

echo "=========================================="
echo "✅ ALL TASKS COMPLETED SUCCESSFULLY!"
echo "=========================================="
