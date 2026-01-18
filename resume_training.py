"""
Resume training from checkpoint with new 6-stage curriculum.

This script loads a trained model checkpoint and resumes training with the
improved curriculum that includes:
- Stage 1.5 (new Stage 2): Easy distance with temporal tolerance
- Gradual difficulty progression through 6 stages

Usage:
    python resume_training.py --checkpoint <path_to_checkpoint>

Example:
    python resume_training.py \
        --checkpoint curriculum_runs/20260103_154409/checkpoints/sac_20260103_154409_150000_steps.zip \
        --replay_buffer curriculum_runs/20260103_154409/checkpoints/sac_20260103_154409_replay_buffer_150000_steps.pkl \
        --start_stage 2 \
        --total_timesteps 500000
"""

import argparse
import os
import time
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
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


def main():
    parser = argparse.ArgumentParser(description='Resume curriculum training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., sac_...150000_steps.zip)')
    parser.add_argument('--replay_buffer', type=str, default=None,
                       help='Path to replay buffer checkpoint (optional but recommended)')
    parser.add_argument('--start_stage', type=int, default=2,
                       help='Which stage to start at (default: 2 for new Stage 1.5)')
    parser.add_argument('--total_timesteps', type=int, default=500000,
                       help='Additional timesteps to train (default: 500K)')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for curriculum logs (default: curriculum_logs_resumed_TIMESTAMP)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save checkpoints (default: curriculum_runs/resumed_TIMESTAMP)')
    parser.add_argument('--checkpoint_freq', type=int, default=50000,
                       help='Save checkpoint every N steps (default: 50000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu, default: cuda)')

    args = parser.parse_args()

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or f"curriculum_logs_resumed_{timestamp}"
    save_dir = args.save_dir or f"curriculum_runs/resumed_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    print("="*70)
    print("RESUME TRAINING WITH IMPROVED 6-STAGE CURRICULUM")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Replay buffer: {args.replay_buffer or 'Not loading'}")
    print(f"Starting stage: {args.start_stage}")
    print(f"Additional timesteps: {args.total_timesteps}")
    print(f"Log directory: {log_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Create environment with NEW 6-stage curriculum
    print("\nInitializing environment with 6-stage curriculum...")
    env = CurriculumCubeTrackingEnv(log_dir=log_dir, enable_curriculum=True)

    # Load model from checkpoint
    print(f"\nLoading model from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = SAC.load(args.checkpoint, env=env, device=args.device)
    print("✓ Model loaded successfully")

    # Load replay buffer if provided
    if args.replay_buffer:
        print(f"\nLoading replay buffer from {args.replay_buffer}...")
        if os.path.exists(args.replay_buffer):
            model.load_replay_buffer(args.replay_buffer)
            print("✓ Replay buffer loaded successfully")
        else:
            print(f"⚠ Warning: Replay buffer not found at {args.replay_buffer}")
            print("  Continuing without replay buffer (will start fresh)")

    # Access the curriculum environment from the vectorized wrapper
    # SAC wraps env in Monitor -> DummyVecEnv
    # Get the actual curriculum env
    actual_env = model.get_env().envs[0].env

    # Manually reset curriculum to desired stage
    print(f"\nResetting curriculum to Stage {args.start_stage}...")
    actual_env.curriculum_manager.current_stage = args.start_stage
    actual_env.curriculum_manager.episode_count = 0
    actual_env.curriculum_manager.success_window.clear()
    print(f"✓ Curriculum reset to Stage {args.start_stage}")
    print(f"  Stage criteria: {actual_env.curriculum_manager.stage_criteria[args.start_stage]}")
    print(f"  Success threshold: {actual_env.curriculum_manager.stage_thresholds[args.start_stage]:.0%}")

    # Setup callbacks
    print("\nSetting up callbacks...")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=f"{save_dir}/checkpoints",
        name_prefix=f"sac_resumed_{timestamp}",
        save_replay_buffer=True,
        save_vecnormalize=False
    )

    # Curriculum monitoring callback (CRITICAL FOR TEMPORAL TOLERANCE VERIFICATION)
    curriculum_callback = CurriculumTrainingCallback(
        log_interval=100,
        verbose=1
    )

    callbacks = CallbackList([checkpoint_callback, curriculum_callback])
    print("✓ Callbacks configured (checkpoint + curriculum monitoring)")

    # Continue training
    print("\n" + "="*70)
    print("STARTING RESUMED TRAINING")
    print("="*70)
    print(f"Training for {args.total_timesteps} additional steps...")
    print(f"Checkpoints will be saved every {args.checkpoint_freq} steps to {save_dir}/checkpoints/")
    print(f"\nMonitor curriculum progress:")
    print(f"  - Check logs in: {log_dir}/")
    print(f"  - Stage transitions: {log_dir}/stage_transitions.csv")
    print("="*70 + "\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=False,  # Continue timestep counting
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )

        # Save final model
        final_model_path = f"{save_dir}/final_model"
        model.save(final_model_path)
        print(f"\n✓ Training completed! Final model saved to: {final_model_path}.zip")

        # Save final replay buffer
        final_buffer_path = f"{save_dir}/final_replay_buffer"
        model.save_replay_buffer(final_buffer_path)
        print(f"✓ Final replay buffer saved to: {final_buffer_path}.pkl")

        # Print curriculum stats
        print("\n" + "="*70)
        print("FINAL CURRICULUM STATUS")
        print("="*70)
        stats = env.get_curriculum_stats()
        print(f"Final stage reached: {stats['stage']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Episodes in current stage: {stats['episode_count']}")
        print(f"Total episodes: {stats['total_episodes']}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        interrupt_model_path = f"{save_dir}/interrupted_model"
        model.save(interrupt_model_path)
        print(f"✓ Model saved to: {interrupt_model_path}.zip")

    finally:
        env.close()
        print("\n✓ Environment closed")


if __name__ == "__main__":
    main()
