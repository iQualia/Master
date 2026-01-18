"""
Run the trained SAC model in PyBullet GUI to visualize the robotic arm's learned behavior.

This script loads the trained curriculum SAC model and runs it in the environment
with visualization enabled.

Usage:
    python run_trained_model.py --model curriculum_runs/stage1_demo/sac_stage1_demo_final.zip
    python run_trained_model.py --checkpoint curriculum_runs/stage1_demo/checkpoints/sac_stage1_demo_20000_steps.zip
"""

import argparse
import time
import numpy as np
from stable_baselines3 import SAC
from curriculum_env import CurriculumCubeTrackingEnv
import pybullet as p
import pybullet_data


def run_trained_model(
    model_path: str,
    num_episodes: int = 5,
    render_mode: str = "gui",
    episode_length: int = 500,
    enable_curriculum: bool = False,
    sleep_time: float = 0.01
):
    """
    Load and run a trained SAC model in the environment.

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        render_mode: "gui" for visualization, "direct" for headless
        episode_length: Maximum steps per episode
        enable_curriculum: Whether to use curriculum stages (usually False for testing)
        sleep_time: Time to sleep between steps for visualization (seconds)
    """
    print("="*80)
    print("RUNNING TRAINED SAC MODEL")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Render Mode: {render_mode}")
    print(f"Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {episode_length}")
    print("="*80)

    # Load the trained model
    print("\nLoading model...")
    model = SAC.load(model_path)
    print("✓ Model loaded successfully")

    # Create environment
    print("\nCreating environment...")
    env = CurriculumCubeTrackingEnv(
        log_dir="eval_logs",
        enable_curriculum=enable_curriculum
    )

    # Switch to GUI mode if requested
    if render_mode == "gui":
        # Disconnect current physics client and reconnect in GUI mode
        p.disconnect(env.unwrapped.physics_client)
        env.unwrapped.physics_client = p.connect(p.GUI)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reload environment objects
        env.unwrapped.plane_id = p.loadURDF("plane.urdf")
        env.unwrapped.robot_id = p.loadURDF("/home/Master/mycobot_urdf_copy.urdf", [-0.1, 0, 0], useFixedBase=True)
        env.unwrapped.cube_id = p.loadURDF("/home/Master/Cube.urdf", [0.3, 0, 0.2])

        # Setup camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.2, 0, 0.2]
        )
        print("✓ Switched to GUI mode")

    print("✓ Environment created")

    # Get robot info
    robot_id = env.unwrapped.robot_id
    num_joints = p.getNumJoints(robot_id)
    print(f"\n📊 Robot Information:")
    print(f"   - Total joints: {num_joints}")
    print(f"   - End-effector link: 6")

    print("\n" + "="*80)
    print("STARTING SIMULATION")
    print("="*80)
    print("\n💡 Controls:")
    print("   - Watch the robot track the red cube")
    print("   - Mouse: Left-drag to rotate, right-drag to pan, scroll to zoom")
    print("   - The robot is using its trained policy (SAC)\n")

    # Run episodes
    episode_rewards = []
    episode_successes = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        # Track episode metrics
        min_distance = float('inf')
        max_alignment = 0.0

        while not done and episode_steps < episode_length:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1

            # Track metrics
            if 'distance_to_cube' in info:
                min_distance = min(min_distance, info['distance_to_cube'])
            if 'alignment' in info:
                max_alignment = max(max_alignment, info['alignment'])

            # Print periodic updates
            if episode_steps % 100 == 0:
                print(f"  Step {episode_steps:3d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Distance: {info.get('distance_to_cube', 0):.4f}m | "
                      f"Alignment: {info.get('alignment', 0):.3f}")

            # Sleep for visualization (only in GUI mode)
            if render_mode == "gui":
                time.sleep(sleep_time)

        # Episode summary
        episode_rewards.append(episode_reward)
        success = info.get('is_success', False)
        episode_successes.append(1.0 if success else 0.0)

        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Steps Taken: {episode_steps}")
        print(f"   Success: {'✓ Yes' if success else '✗ No'}")
        print(f"   Min Distance: {min_distance:.4f}m")
        print(f"   Max Alignment: {max_alignment:.3f}")
        print(f"   Final Distance: {info.get('distance_to_cube', 0):.4f}m")

    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success Rate: {np.mean(episode_successes):.1%}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print("="*80)

    # Close environment
    env.close()
    print("\n✓ Simulation complete!")


def main():
    parser = argparse.ArgumentParser(description="Run trained SAC model in PyBullet")

    parser.add_argument(
        "--model",
        type=str,
        default="curriculum_runs/stage1_demo/sac_stage1_demo_final.zip",
        help="Path to trained model file"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (headless mode)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.01,
        help="Sleep time between steps in seconds (default: 0.01, use 0 for max speed)"
    )

    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum stages during testing"
    )

    args = parser.parse_args()

    # Determine render mode
    render_mode = "direct" if args.headless else "gui"

    # Run the model
    run_trained_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render_mode=render_mode,
        episode_length=args.steps,
        enable_curriculum=args.curriculum,
        sleep_time=args.speed
    )


if __name__ == "__main__":
    main()
