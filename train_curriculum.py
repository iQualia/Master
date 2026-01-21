"""
Main training script for curriculum-based SAC learning.

Integrates:
- CurriculumCubeTrackingEnv (5-stage progressive curriculum)
- SAC algorithm from stable-baselines3
- Custom callbacks for curriculum stage tracking
- TensorBoard logging with curriculum metrics

Usage:
    python train_curriculum.py --total_timesteps 500000 --log_dir curriculum_runs/run_1
"""

import argparse
import time
import os
from datetime import datetime
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from curriculum_env import CurriculumCubeTrackingEnv


class CurriculumTrainingCallback(BaseCallback):
    """
    Custom callback for curriculum learning that logs:
    - Curriculum stage transitions
    - Stage-specific success rates
    - Reward component breakdown
    - Episode statistics
    """

    def __init__(self, log_interval: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"CURRICULUM TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.model.device}")
        print(f"{'='*80}\n")

    def _on_step(self) -> bool:
        # Get info from environment
        infos = self.locals.get("infos")
        if infos and isinstance(infos, list):
            info = infos[0]  # Single environment

            # Track episode completion
            if self.locals.get("dones")[0]:
                self.episode_count += 1

                # Log curriculum stats to TensorBoard
                if 'curriculum_stage' in info:
                    stage = info['curriculum_stage']
                    success_rate = info.get('success_rate', 0.0)

                    self.logger.record("curriculum/stage", stage)
                    self.logger.record("curriculum/success_rate", success_rate)

                    # Log reward components if available
                    if 'reward_components' in info:
                        components = info['reward_components']
                        for component_name, value in components.items():
                            self.logger.record(f"reward/{component_name}", value)

                    # Log environment metrics
                    self.logger.record("env/distance_to_cube", info.get('distance_to_cube', 0))
                    self.logger.record("env/alignment", info.get('alignment', 0))
                    self.logger.record("env/cube_visible", 1.0 if info.get('cube_visible', False) else 0.0)

            # Periodic console logging
            if self.n_calls % self.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                stage = info.get('curriculum_stage', 'N/A')
                success_rate = info.get('success_rate', 0.0)

                if self.verbose > 0:
                    print(f"Step: {self.n_calls:7d} | "
                          f"Time: {elapsed_time:6.1f}s | "
                          f"Stage: {stage} | "
                          f"Success: {success_rate:5.1%} | "
                          f"Distance: {info.get('distance_to_cube', 0):.4f}")

        return True


def make_curriculum_env(log_dir: str, enable_curriculum: bool = True, restore_curriculum: str = None):
    """
    Create curriculum environment wrapper.

    Args:
        log_dir: Directory for curriculum logs
        enable_curriculum: If False, runs baseline C1.py (for comparison)
        restore_curriculum: Path to curriculum_state.json to restore stage progress

    Returns:
        Wrapped environment ready for SAC training
    """
    def _init():
        env = CurriculumCubeTrackingEnv(
            log_dir=os.path.join(log_dir, "curriculum_logs"),
            enable_curriculum=enable_curriculum,
            restore_curriculum=restore_curriculum
        )
        return env
    return _init


def train_curriculum(
    total_timesteps: int = 500000,
    log_dir: str = "curriculum_runs",
    run_name: str = None,
    enable_curriculum: bool = True,
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,  # Reduced from 1M for 4GB GPU
    batch_size: int = 256,
    checkpoint_freq: int = 10000,
    device: str = "cuda",
    resume_from: str = None,
    restore_curriculum: str = None
):
    """
    Train SAC with curriculum learning.

    Args:
        total_timesteps: Total training steps
        log_dir: Base directory for logs and checkpoints
        run_name: Name for this training run (auto-generated if None)
        enable_curriculum: If False, trains baseline without curriculum
        learning_rate: SAC learning rate
        buffer_size: Replay buffer size (reduced for laptop GPU)
        batch_size: Training batch size
        checkpoint_freq: Save model every N steps
        device: 'cuda' or 'cpu'
        resume_from: Path to checkpoint to resume from
        restore_curriculum: Path to curriculum_state.json to restore stage progress
    """
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "curriculum" if enable_curriculum else "baseline"
        run_name = f"{mode}_{timestamp}"

    # Create directories
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tensorboard_log = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run Name: {run_name}")
    print(f"Mode: {'Curriculum Learning' if enable_curriculum else 'Baseline (C1.py)'}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Buffer Size: {buffer_size:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"TensorBoard Log: {tensorboard_log}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*80}\n")

    # Create environment
    env = DummyVecEnv([make_curriculum_env(run_dir, enable_curriculum, restore_curriculum)])

    # Create or load model
    if resume_from:
        print(f"Loading model from: {resume_from}")
        model = SAC.load(
            resume_from,
            env=env,
            device=device,
            tensorboard_log=tensorboard_log
        )
        print(f"✓ Model loaded successfully")

        # Try to load replay buffer (critical for SAC performance)
        import re
        match = re.search(r'(\d+)_steps\.zip$', resume_from)
        if match:
            steps = match.group(1)
            # Correct path construction: sac_XXX_replay_buffer_300000_steps.pkl
            checkpoint_dir = os.path.dirname(resume_from)
            checkpoint_name = os.path.basename(resume_from)
            buffer_name = checkpoint_name.replace(f'_{steps}_steps.zip', f'_replay_buffer_{steps}_steps.pkl')
            buffer_path = os.path.join(checkpoint_dir, buffer_name)

            if os.path.exists(buffer_path):
                print(f"Loading replay buffer from: {buffer_path}")
                model.load_replay_buffer(buffer_path)
                print(f"✓ Replay buffer loaded successfully\n")
            else:
                print(f"[WARNING] Replay buffer not found at: {buffer_path}")
                print("Training will continue with empty buffer (performance may be degraded)\n")
        else:
            print("[WARNING] Could not parse step count from checkpoint filename")
            print("Replay buffer not loaded\n")
    else:
        # SAC hyperparameters optimized for 4GB GPU
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",  # Auto-tune entropy
            target_update_interval=1,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            # CRITICAL for 4GB GPU: offload buffer to CPU RAM
            replay_buffer_kwargs=dict(
                handle_timeout_termination=False
            )
        )

        # Try to move buffer to CPU to save GPU memory
        if hasattr(model, 'replay_buffer') and device == "cuda":
            print("⚠️  Note: Replay buffer on GPU. If OOM occurs, manually set buffer_device='cpu'\n")

    # Create callbacks
    curriculum_callback = CurriculumTrainingCallback(
        log_interval=100,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"sac_{run_name}",
        save_replay_buffer=True,  # Save buffer for resuming
        save_vecnormalize=True
    )

    # Train model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, checkpoint_callback],
            log_interval=10,
            tb_log_name=run_name,
            reset_num_timesteps=(resume_from is None)  # Only reset if not resuming
        )
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user!")
        print("Saving current model state...")
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {type(e).__name__}: {e}")
        print("Attempting emergency save...")
        try:
            emergency_path = os.path.join(run_dir, f"sac_{run_name}_emergency.zip")
            model.save(emergency_path)
            print(f"[OK] Emergency model saved to: {emergency_path}")
        except Exception as save_error:
            print(f"[ERROR] Emergency save failed: {save_error}")
        raise

    # Save final model
    final_model_path = os.path.join(run_dir, f"sac_{run_name}_final.zip")
    model.save(final_model_path)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Final model saved to: {final_model_path}")
    print(f"TensorBoard logs: tensorboard --logdir {tensorboard_log}")
    print(f"{'='*80}\n")

    # Get final curriculum stats from environment
    if enable_curriculum:
        # env.envs[0] is the CurriculumCubeTrackingEnv (gym.Wrapper)
        curriculum_env = env.envs[0]
        stats = None
        if hasattr(curriculum_env, 'get_curriculum_stats'):
            try:
                stats = curriculum_env.get_curriculum_stats()
            except Exception as e:
                print(f"[WARNING] Could not get curriculum stats: {e}")

        if stats:
            print(f"\nFinal Curriculum Statistics:")
            print(f"  Final Stage Reached: {stats['stage']}")
            print(f"  Total Episodes: {stats['total_episodes']}")
            print(f"  Final Success Rate: {stats['success_rate']:.2%}")
            print(f"  Episodes in Final Stage: {stats['episode_count']}")

    env.close()

    return model, run_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAC with curriculum learning")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps (default: 500k)")
    parser.add_argument("--log_dir", type=str, default="curriculum_runs",
                        help="Base directory for logs (default: curriculum_runs)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (auto-generated if not provided)")

    # Curriculum control
    parser.add_argument("--baseline", action="store_true",
                        help="Disable curriculum (run baseline C1.py)")

    # SAC hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Replay buffer size (default: 100k for 4GB GPU)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")

    # Checkpointing
    parser.add_argument("--checkpoint_freq", type=int, default=10000,
                        help="Save checkpoint every N steps (default: 10k)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--restore_curriculum", type=str, default=None,
                        help="Path to curriculum_state.json to restore stage progress")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Training device (default: cuda)")

    args = parser.parse_args()

    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print GPU info if using CUDA
    if args.device == "cuda":
        print(f"\n🎮 GPU Information:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Free Memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB\n")

    # Run training
    model, run_dir = train_curriculum(
        total_timesteps=args.total_timesteps,
        log_dir=args.log_dir,
        run_name=args.run_name,
        enable_curriculum=(not args.baseline),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        resume_from=args.resume_from,
        restore_curriculum=args.restore_curriculum
    )

    print(f"\n✓ Training completed successfully!")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
