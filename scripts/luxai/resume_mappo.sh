#!/bin/bash
# Script to resume MAPPO training from the 1.95M checkpoint up to 6M frames

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../BenchMARL"

# Load virtual environment
source ../.venv/bin/activate

MAPPO_CKPT="/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL/outputs/2026-04-27/09-48-59/mappo_match_v2_cnn__d20a96a5_26_04_27-09_48_59/checkpoints/checkpoint_1950000.pt"

echo "=========================================="
echo "🚀 RESUMING MAPPO (From 1.95M to 6M frames)..."
echo "=========================================="

XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=mappo task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.restore_file=$MAPPO_CKPT \
    experiment.max_n_frames=6000000 experiment.exploration_eps_end=0.05 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.on_policy_n_envs_per_worker=4 experiment.on_policy_minibatch_size=100 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=40 \
    experiment.exclude_buffer_from_checkpoint=true \
    experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN_6M" \
    experiment.wandb_extra_kwargs.name="mappo_16ch_biggercnn_6M_resumed"

echo "=========================================="
echo "✅ MAPPO RESUME TRAINING COMPLETED"
echo "=========================================="
