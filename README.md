# Curriculum Learning for Visual Servoing

Reinforcement learning project for training a MyCobot robot arm to perform visual servoing tasks using curriculum learning with SAC (Soft Actor-Critic).

## Quick Start (After Cloning)

```bash
# 1. Clone the repository
git clone https://github.com/iQualia/Master.git
cd Master

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements_minimal.txt

# 4. Verify setup
python -c "from C1 import CubeTrackingEnv; env = CubeTrackingEnv(); env.reset(); print('Environment OK')"

# 5. Run baseline evaluation (verify reward functions work)
python evaluate_baselines.py --mode headless --episodes 10 --stages 1 2 3 4 5 6

# 6. Start curriculum training
python train_curriculum.py --total_timesteps 500000 --run_name "my_training_run"
```

---

## Directory Structure

```
Pybullet/
├── C1.py                      # Core environment (CubeTrackingEnv)
├── curriculum_env.py          # Curriculum wrapper (CurriculumCubeTrackingEnv)
├── curriculum_manager.py      # 6-stage progression logic
├── reward_functions.py        # Reward function definitions
│
├── train_curriculum.py        # Main training script (single env)
├── train_curriculum_fast.py   # Vectorized training (for HPC)
├── resume_training.py         # Resume interrupted training
│
├── baseline_policies.py       # Random, Scripted, Oracle policies
├── evaluate_baselines.py      # Baseline evaluation harness
├── run_trained_model.py       # Run trained SAC model
├── plot_baselines.py          # Generate evaluation plots
│
├── mycobot_urdf.urdf          # Robot URDF (7 joints)
├── mycobot_urdf_copy.urdf     # Backup URDF
├── Cube.urdf                  # Target cube
├── Inspection_Station.urdf    # Environment station
├── meshes/                    # Robot joint meshes (.dae, .png)
│   ├── joint1.dae ... joint7.dae
│   └── joint1.png ... joint7.png
│
├── utils/
│   ├── __init__.py
│   └── episode_logger.py      # CSV episode logging
│
├── requirements_minimal.txt   # Python dependencies
├── .gitignore                 # Excludes outputs, venv, pycache
│
├── train_with_resume.sh       # Training with auto-resume
├── monitor_training.sh        # TensorBoard monitoring
│
├── CURRICULUM_README.md       # Detailed curriculum docs
├── HPC_DEPLOYMENT_GUIDE.md    # HPC/Cloud deployment guide
└── README.md                  # This file
```

---

## Curriculum Stages

The training uses 6 progressive stages:

| Stage | Reward Components | Success Criteria |
|-------|-------------------|------------------|
| 1 | Visibility only (+10/-10) | visibility=True |
| 2 | Visibility + Easy Distance | distance < 0.25m |
| 3 | Visibility + Medium Distance | distance < 0.20m |
| 4 | + Partial Alignment (weight=5) | distance < 0.15m, align > 0.3 |
| 5 | + Full Alignment (weight=10) | distance < 0.12m, align > 0.5 |
| 6 | Sparse Rewards | distance < 0.10m, align > 0.6 |

Stage advancement occurs when success rate reaches threshold (70%→65%→60%→50%→40%→35%).

---

## Key Commands

### Training

```bash
# Standard training (500k steps)
python train_curriculum.py --total_timesteps 500000 --run_name "run_v1"

# Vectorized training (faster, needs more RAM)
python train_curriculum_fast.py --total_timesteps 500000 --n_envs 4

# Resume interrupted training
python resume_training.py --checkpoint curriculum_runs/run_v1/checkpoints/latest.zip
```

### Evaluation

```bash
# Full baseline evaluation
python evaluate_baselines.py --mode headless --episodes 10 --stages 1 2 3 4 5 6

# Run trained model (GUI)
python run_trained_model.py --model curriculum_runs/run_v1/final_model.zip --mode gui

# Generate plots
python plot_baselines.py --input baseline_eval
```

### Monitoring

```bash
# TensorBoard (in separate terminal)
tensorboard --logdir curriculum_runs/run_v1/tensorboard

# Key metrics to watch:
# - curriculum/stage: Current curriculum stage
# - curriculum/success_rate: Rolling success rate
# - rollout/ep_rew_mean: Mean episode reward
```

---

## Environment Details

**Observation Space** (14D continuous):
- Joint positions (7D)
- Joint velocities (7D)
- Cube position MASKED when not visible (set to [0,0,0])

**Action Space** (7D continuous):
- Joint velocity commands [-1, 1] for each joint

**Episode**:
- Length: 300 steps
- Termination: Step limit or collision

**Robot**:
- MyCobot 280 (7 DOF)
- End-effector mounted camera
- Base at [-0.1, 0, 0]

**Cube**:
- Spawns randomly at [0.25-0.32, -0.06-0.06, 0.1]
- Distance ~0.35-0.42m from robot base

---

## Baseline Results (Expected)

After running `evaluate_baselines.py`:

| Policy | Stage 6 Median Return | Visibility |
|--------|----------------------|------------|
| Oracle | ~2990 | ~100% |
| Scripted | ~2990 | ~100% |
| Random | ~2500 | ~65-90% |

Oracle >= Scripted > Random validates reward shaping is correct.

---

## Files Explained

### Core Files

- **C1.py**: Base Gym environment. Handles PyBullet simulation, observations, rewards, camera rendering. The `CubeTrackingEnv` class is the foundation.

- **curriculum_env.py**: Wraps `CubeTrackingEnv` with curriculum logic. Dynamically adjusts rewards based on current stage.

- **curriculum_manager.py**: Tracks success rates, manages stage transitions. Key parameters: `stage_criteria`, `stage_thresholds`, `min_episodes_per_stage_map`.

- **reward_functions.py**: Modular reward components (visibility, distance, alignment). Used by curriculum_env.

### Training Files

- **train_curriculum.py**: Main entry point. Creates SAC agent, runs training loop with curriculum callbacks.

- **train_curriculum_fast.py**: Uses `SubprocVecEnv` for parallel environments. Faster but needs more memory.

### Evaluation Files

- **baseline_policies.py**: Three baseline policies:
  - `RandomPolicy`: Uniform random actions
  - `ScriptedPolicy`: Random search + IK when cube visible
  - `OraclePolicy`: Uses privileged true cube position + IK

- **evaluate_baselines.py**: Runs all baselines across all stages, generates CSV logs and summary.

---

## Troubleshooting

### "No module named pybullet"
```bash
pip install pybullet
```

### "Failed to load URDF"
Ensure meshes/ directory exists with all .dae files.

### Training stuck at Stage 1
Check `curriculum_runs/*/curriculum_logs/stage_transitions.csv` for success rate. May need more episodes or threshold adjustment.

### Low visibility (<50%)
Robot may not be initializing to correct pose. Check `C1.py` reset() function.

---

## Output Directories (Gitignored)

These are generated during training/evaluation and excluded from git:

```
curriculum_runs/          # Training checkpoints, logs, models
baseline_eval/            # Evaluation results
sac_tensorboard/          # TensorBoard logs
*.pkl                     # Pickle files
```

---

## Dependencies (requirements_minimal.txt)

```
torch==2.5.1
stable-baselines3==2.3.0
gymnasium==0.29.1
gym==0.21.0
pybullet==3.2.6
tensorboard==2.18.0
matplotlib==3.8.0
pandas==2.1.0
numpy==1.26.0
opencv-python==4.8.0
```

Install with:
```bash
pip install -r requirements_minimal.txt
```

---

## Next Steps After Training

1. Training completes at 500k steps
2. Model saved to `curriculum_runs/run_name/final_model.zip`
3. Evaluate: `python run_trained_model.py --model path/to/model.zip --mode gui`
4. Check TensorBoard for learning curves
5. Compare to baselines in thesis

---

## Contact

Repository: https://github.com/iQualia/Master.git
