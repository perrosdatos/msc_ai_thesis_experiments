#!/bin/bash
# Script to resume MASAC training from the 4.05M checkpoint up to 6M frames

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../BenchMARL"

# Load virtual environment
source ../.venv/bin/activate

MASAC_CKPT="/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL/outputs/2026-04-27/22-59-06/masac_match_v2_cnn__5490d15d_26_04_27-22_59_06/checkpoints/checkpoint_4050000.pt"

echo "=========================================="
echo "🚀 RESUMING MASAC (From 4.05M to 6M frames)..."
echo "=========================================="

XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=masac task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.restore_file=$MASAC_CKPT \
    experiment.max_n_frames=6000000 experiment.exploration_eps_end=0.05 \
    experiment.sampling_device=cpu experiment.train_device=cuda experiment.buffer_device=cpu \
    experiment.evaluation_episodes=3 experiment.off_policy_collected_frames_per_batch=1000 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=45 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=40 \
    experiment.exclude_buffer_from_checkpoint=true \
    experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN_6M" \
    experiment.wandb_extra_kwargs.name="masac_16ch_biggercnn_6M_resumed"

echo "=========================================="
echo "✅ MASAC RESUME TRAINING COMPLETED"
echo "=========================================="
