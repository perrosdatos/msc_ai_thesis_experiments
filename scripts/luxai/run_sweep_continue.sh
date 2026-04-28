#!/bin/bash
# Continuation training script for MAPPO and MASAC
# Starts from the specified checkpoints (3M frames) and continues up to 6M frames.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../BenchMARL"

# Load virtual environment
source ../.venv/bin/activate

# Use absolute paths for checkpoints to avoid resolution issues
MASAC_CKPT="$(pwd)/outputs/2026-04-24/04-47-15/masac_match_v2_cnn__d0cf953a_26_04_24-04_47_15/checkpoints/checkpoint_3000000.pt"
MAPPO_CKPT="$(pwd)/outputs/2026-04-22/09-58-15/mappo_match_v2_cnn__6d9aeb28_26_04_22-09_58_15/checkpoints/checkpoint_3000000.pt"

echo "=========================================="
echo "🚀 CONTINUING MASAC (From 3M to 6M frames)..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=masac task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.restore_file=$MASAC_CKPT \
    experiment.max_n_frames=6000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda experiment.buffer_device=cpu \
    experiment.evaluation_episodes=3 experiment.off_policy_collected_frames_per_batch=1000 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=45 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=20 experiment.exclude_buffer_from_checkpoint=true experiment.project_name="Lux_Thesis_16CH_v2_Continued" experiment.wandb_extra_kwargs.name="masac_16ch_continued_3M_to_6M" +experiment.wandb_extra_kwargs.id="masac_16ch_continued_3M_to_6M"

echo "=========================================="
echo "🚀 CONTINUING MAPPO (From 3M to 6M frames)..."
echo "=========================================="

XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=mappo task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.restore_file=$MAPPO_CKPT \
    experiment.max_n_frames=6000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.on_policy_n_envs_per_worker=4 experiment.on_policy_minibatch_size=100 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=20 experiment.exclude_buffer_from_checkpoint=true experiment.project_name="Lux_Thesis_16CH_v2_Continued" experiment.wandb_extra_kwargs.name="mappo_16ch_continued_3M_to_6M" +experiment.wandb_extra_kwargs.id="mappo_16ch_continued_3M_to_6M"

echo "=========================================="
echo "✅ CONTINUATION TRAINING COMPLETED"
echo "=========================================="
