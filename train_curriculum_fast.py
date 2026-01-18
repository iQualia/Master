"""
FAST curriculum training with vectorized environments.
Uses multiple parallel environments for 5-10x speedup.

For HPC deployment with GPU clusters.
"""

import argparse
import time
import os
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch

from curriculum_env import CurriculumCubeTrackingEnv


def make_env(log_dir, enable_curriculum=True, rank=0, seed=0):
    """
    Utility function for multiprocessed env.

    Args:
        log_dir: Directory for logs
        enable_curriculum: Enable curriculum learning
        rank: Index of the subprocess
        seed: Random seed
    """
    def _init():
        env = CurriculumCubeTrackingEnv(
            log_dir=os.path.join(log_dir, f"curriculum_logs_env{rank}"),
            enable_curriculum=enable_curriculum
        )
        # env.seed(seed + rank)  # Removed: Gymnasium envs don't use seed() method
        return env
    return _init


class CurriculumTrainingCallback(BaseCallback):
    """Fast callback with reduced logging overhead."""

    def __init__(self, log_interval: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"FAST CURRICULUM TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Vectorized Envs: {self.training_env.num_envs}")
        print(f"Device: {self.model.device}")
        print(f"{'='*80}\n")

    def _on_step(self) -> bool:
        # Only log curriculum stats from first environment
        if self.n_calls % self.log_interval == 0:
            infos = self.locals.get("infos")
            if infos and len(infos) > 0:
                info = infos[0]  # First env

                if 'curriculum_stage' in info:
                    self.logger.record("curriculum/stage", info['curriculum_stage'])
                    self.logger.record("curriculum/success_rate", info.get('success_rate', 0.0))

                    if self.verbose > 0:
                        elapsed = time.time() - self.start_time
                        print(f"Steps: {self.n_calls:8d} | "
                              f"Time: {elapsed:6.1f}s | "
                              f"Stage: {info['curriculum_stage']} | "
                              f"Success: {info.get('success_rate', 0.0):5.1%}")

        return True


def train_curriculum_fast(
    total_timesteps: int = 1000000,
    log_dir: str = "curriculum_runs_fast",
    run_name: str = None,
    n_envs: int = 8,  # Number of parallel environments
    enable_curriculum: bool = True,
    learning_rate: float = 3e-4,
    buffer_size: int = 200000,  # Larger for HPC
    batch_size: int = 512,  # Larger batches for GPU efficiency
    checkpoint_freq: int = 100000,
    device: str = "cuda",
    use_subproc: bool = True,  # Use subprocesses (faster on HPC)
    resume_from: str = None,
    seed: int = 0
):
    """
    Train SAC with curriculum learning using vectorized environments.

    Args:
        n_envs: Number of parallel environments (4-16 recommended)
        use_subproc: Use SubprocVecEnv (faster) vs DummyVecEnv (easier to debug)
        Other args: Same as train_curriculum.py
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "curriculum" if enable_curriculum else "baseline"
        run_name = f"{mode}_fast_{n_envs}envs_{timestamp}"

    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tensorboard_log = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"FAST TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run Name: {run_name}")
    print(f"Mode: {'Curriculum' if enable_curriculum else 'Baseline'}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs: {n_envs} ({'SubprocVecEnv' if use_subproc else 'DummyVecEnv'})")
    print(f"Expected Speedup: ~{n_envs}x over single env")
    print(f"Learning Rate: {learning_rate}")
    print(f"Buffer Size: {buffer_size:,}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Create vectorized environment
    if use_subproc and n_envs > 1:
        # SubprocVecEnv: Each env in separate process (faster, better for HPC)
        env = SubprocVecEnv([
            make_env(run_dir, enable_curriculum, i, seed)
            for i in range(n_envs)
        ])
        print(f"✓ Created {n_envs} parallel environments (SubprocVecEnv)")
    else:
        # DummyVecEnv: Sequential execution (easier debugging)
        env = DummyVecEnv([
            make_env(run_dir, enable_curriculum, i, seed)
            for i in range(n_envs)
        ])
        print(f"✓ Created {n_envs} environments (DummyVecEnv)")

    # Create or load model
    if resume_from:
        print(f"Loading model from: {resume_from}")
        model = SAC.load(resume_from, env=env, device=device)
        print(f"✓ Model loaded\n")
    else:
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
            ent_coef="auto",
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
            # Larger network for better final performance on cloud V100
            policy_kwargs=dict(net_arch=[512, 512, 256])  # Deeper network for Stage 5 precision
        )

    # Callbacks
    curriculum_callback = CurriculumTrainingCallback(
        log_interval=100,  # More frequent logging for progress visibility
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,  # Adjust for vectorized envs
        save_path=checkpoint_dir,
        name_prefix=f"sac_{run_name}",
        save_replay_buffer=True
    )

    # Train
    try:
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, checkpoint_callback],
            log_interval=100,
            tb_log_name=run_name,
            reset_num_timesteps=(resume_from is None)
        )

        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Steps/second: {total_timesteps/total_time:.1f}")
        print(f"Speedup vs single env: ~{(total_timesteps/total_time)/4.5:.1f}x")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")

    # Save final model
    final_model_path = os.path.join(run_dir, f"sac_{run_name}_final.zip")
    model.save(final_model_path)
    print(f"Final model: {final_model_path}")

    env.close()
    return model, run_dir


def main():
    parser = argparse.ArgumentParser(description="Fast curriculum training with vectorized envs")

    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--log_dir", type=str, default="curriculum_runs_fast")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments (4-16 recommended)")
    parser.add_argument("--baseline", action="store_true")

    # SAC hyperparameters (optimized for HPC)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=200000,
                        help="Larger buffer for HPC (200k-500k)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Larger batches for GPU efficiency")

    parser.add_argument("--checkpoint_freq", type=int, default=100000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dummy_vec", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv (for debugging)")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Document configuration for reproducibility
    from document_config import document_training_run
    import os
    run_dir = f"{args.log_dir}/{args.run_name if args.run_name else 'latest'}"
    os.makedirs(run_dir, exist_ok=True)
    document_training_run(run_dir, args)

    train_curriculum_fast(
        total_timesteps=args.total_timesteps,
        log_dir=args.log_dir,
        run_name=args.run_name,
        n_envs=args.n_envs,
        enable_curriculum=(not args.baseline),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        use_subproc=(not args.dummy_vec),
        resume_from=args.resume_from,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
