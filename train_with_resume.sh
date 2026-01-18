#!/bin/bash
# Automatic checkpoint resume for spot instance interruptions

RUN_NAME="${1:-curriculum_1M_run1}"
TOTAL_STEPS="${2:-1000000}"
CHECKPOINT_DIR="curriculum_runs_fast/${RUN_NAME}/checkpoints"

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/sac_*.zip 2>/dev/null | grep -v replay | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "✓ Resuming from: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_from $LATEST_CHECKPOINT"
else
    echo "Starting fresh training"
    RESUME_ARG=""
fi

# Launch training
python train_curriculum_fast.py \
    --total_timesteps $TOTAL_STEPS \
    --run_name $RUN_NAME \
    --n_envs 8 \
    --buffer_size 300000 \
    --batch_size 512 \
    --checkpoint_freq 100000 \
    --device cuda \
    $RESUME_ARG \
    2>&1 | tee "training_${RUN_NAME}.log"
