#!/bin/bash
# Standalone training script for QMIX (V2 - 16 Channels) up to 6M frames

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../BenchMARL"

# Load virtual environment
source ../.venv/bin/activate

echo "=========================================="
echo "🚀 STARTING QMIX (6M Frames)..."
echo "=========================================="

XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=qmix task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.max_n_frames=6000000 experiment.exploration_eps_end=0.05 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.off_policy_collected_frames_per_batch=400 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=20 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=40 \
    experiment.exclude_buffer_from_checkpoint=true \
    experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN_6M" \
    experiment.wandb_extra_kwargs.name="qmix_16ch_biggercnn_6M_run"

echo "=========================================="
echo "✅ QMIX TRAINING COMPLETED"
echo "=========================================="
