#!/bin/bash
# Training monitoring script

echo "================================================================================"
echo "SAC TRAINING MONITOR"
echo "================================================================================"
echo ""

# Check if training is running
if pgrep -f "train_curriculum.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    PID=$(pgrep -f "train_curriculum.py")
    echo "  Process ID: $PID"
    echo ""
else
    echo "✗ Training is NOT running"
    echo ""
    exit 1
fi

# Show latest progress
echo "Latest Training Progress:"
echo "--------------------------------------------------------------------------------"
tail -50 /home/iqraq/Reinforcement_Learning/Pybullet/training_100k.log | \
    grep -E "Step:|total_timesteps|actor_loss|Success:|Stage:" | tail -10
echo "--------------------------------------------------------------------------------"
echo ""

# Show curriculum stats
echo "Curriculum Progress:"
echo "--------------------------------------------------------------------------------"
tail -100 /home/iqraq/Reinforcement_Learning/Pybullet/training_100k.log | \
    grep -E "Step.*Stage.*Success" | tail -5
echo "--------------------------------------------------------------------------------"
echo ""

# Count checkpoints
CHECKPOINTS=$(ls -1 /home/iqraq/Reinforcement_Learning/Pybullet/curriculum_runs/stage_100k/checkpoints/*.zip 2>/dev/null | wc -l)
echo "Checkpoints saved: $CHECKPOINTS"
echo ""

# Estimate progress
CURRENT_STEP=$(tail -100 /home/iqraq/Reinforcement_Learning/Pybullet/training_100k.log | \
    grep "total_timesteps" | tail -1 | awk '{print $4}')
if [ ! -z "$CURRENT_STEP" ]; then
    PROGRESS=$(echo "scale=1; $CURRENT_STEP / 1000" | bc)
    echo "Current step: ${CURRENT_STEP} / 100,000 (${PROGRESS}%)"

    REMAINING=$(echo "100000 - $CURRENT_STEP" | bc)
    TIME_PER_10K=38  # minutes
    EST_MINUTES=$(echo "scale=0; $REMAINING * $TIME_PER_10K / 10000" | bc)
    EST_HOURS=$(echo "scale=1; $EST_MINUTES / 60" | bc)
    echo "Estimated time remaining: ~${EST_HOURS} hours"
fi

echo ""
echo "================================================================================"
echo "Commands:"
echo "  View live log:        tail -f training_100k.log"
echo "  Stop training:        pkill -f train_curriculum.py"
echo "  TensorBoard:          tensorboard --logdir curriculum_runs/stage_100k/tensorboard"
echo "================================================================================"
