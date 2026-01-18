"""
Curriculum Learning Wrapper for CubeTrackingEnv.

Wraps the existing C1.py environment to implement 6-stage progressive curriculum:
- Stage 1: Visibility only
- Stage 2: Visibility + Easy Distance (0.22m) with temporal tolerance - NEW
- Stage 3: Visibility + Medium Distance (0.18m)
- Stage 4: Visibility + Distance + Partial Alignment
- Stage 5: Visibility + Distance + Full Alignment
- Stage 6: Sparse rewards for final policy

Automatically advances through stages based on success criteria.
"""

import gym
from gym import spaces
import numpy as np
from collections import deque
from typing import Dict, Tuple
import os
import sys

# Import the base environment from C1.py
from C1 import CubeTrackingEnv

# Import curriculum components
from curriculum_manager import CurriculumManager
from reward_functions import (
    compute_stage1_reward,
    compute_stage2_reward,  # NEW: Temporal tolerance
    compute_stage3_reward,
    compute_stage4_reward,
    compute_stage5_reward,
    compute_stage6_reward
)


class CurriculumCubeTrackingEnv(gym.Wrapper):
    """
    Curriculum learning wrapper around CubeTrackingEnv.

    Overrides reward computation and tracks episode success to enable
    automatic curriculum progression.
    """

    def __init__(self, log_dir: str = "curriculum_logs", enable_curriculum: bool = True, mode: str = 'headless'):
        """
        Initialize curriculum wrapper.

        Args:
            log_dir: Directory for curriculum transition logs
            enable_curriculum: If False, runs baseline C1.py behavior (for comparison)
            mode: 'gui' for visual verification, 'headless' for speed
        """
        # Initialize base environment
        base_env = CubeTrackingEnv(mode=mode)
        super().__init__(base_env)

        # Curriculum manager
        self.curriculum_manager = CurriculumManager(log_dir=log_dir)
        self.enable_curriculum = enable_curriculum

        # Episode tracking
        self.episode_distances = []
        self.episode_alignments = []
        self.episode_visibility_frames = 0
        self.episode_total_frames = 0

        # Temporal tracking for Stage 2 temporal tolerance
        self.prev_distance = None
        self.visibility_history = deque(maxlen=10)  # Track last 10 steps

        # Map stage to reward function
        self.reward_functions = {
            1: compute_stage1_reward,
            2: compute_stage2_reward,  # NEW: Temporal tolerance
            3: compute_stage3_reward,
            4: compute_stage4_reward,
            5: compute_stage5_reward,
            6: compute_stage6_reward
        }

        print(f"\n{'='*70}")
        print(f"CURRICULUM LEARNING ENVIRONMENT INITIALIZED")
        print(f"{'='*70}")
        print(f"Starting Stage: {self.curriculum_manager.get_current_stage()}")
        print(f"Curriculum Enabled: {self.enable_curriculum}")
        print(f"Log Directory: {log_dir}")
        print(f"{'='*70}\n")

    def reset(self):
        """
        Reset environment and check for curriculum advancement.

        Returns:
            observation: Initial observation from environment
        """
        # Check if should advance curriculum (before resetting episode tracking)
        if self.enable_curriculum and self.episode_total_frames > 0:
            # Calculate episode success based on current stage criteria
            avg_distance = np.mean(self.episode_distances) if self.episode_distances else float('inf')
            avg_alignment = np.mean(self.episode_alignments) if self.episode_alignments else -1.0
            visibility_ratio = self.episode_visibility_frames / max(self.episode_total_frames, 1)

            # Get current stage criteria
            stage_config = self.curriculum_manager.get_stage_config()
            criteria = stage_config['success_criteria']

            # Check success based on stage-specific criteria
            success = True

            # All stages require visibility
            if 'visibility' in criteria and criteria['visibility']:
                # For stage 1, require 70% visibility
                required_visibility = 0.70 if self.curriculum_manager.get_current_stage() == 1 else 0.50
                success = success and (visibility_ratio >= required_visibility)

            # Check distance if required
            if 'distance' in criteria:
                success = success and (avg_distance < criteria['distance'])

            # Check alignment if required
            if 'alignment' in criteria:
                success = success and (avg_alignment > criteria['alignment'])

            # Record episode outcome
            self.curriculum_manager.record_episode(
                success=success,
                distance=avg_distance,
                alignment=avg_alignment
            )

            # Check for stage advancement
            if self.curriculum_manager.should_advance_stage():
                self.curriculum_manager.advance_stage()

            # Periodic logging
            self.curriculum_manager.periodic_log()

        # Reset episode tracking
        self.episode_distances = []
        self.episode_alignments = []
        self.episode_visibility_frames = 0
        self.episode_total_frames = 0

        # Reset temporal tracking
        self.prev_distance = None
        self.visibility_history.clear()

        # Reset base environment
        obs = self.env.reset()

        # Return just obs for Stable-Baselines3 compatibility
        # (SB3 with SubprocVecEnv doesn't support new Gymnasium API yet)
        return obs

    def step(self, action):
        """
        Execute action and compute curriculum-aware reward.

        Args:
            action: Action to execute in environment

        Returns:
            observation: Next state observation
            reward: Curriculum-adjusted reward
            terminated: Episode termination flag (goal reached or failure)
            truncated: Episode truncation flag (time limit)
            info: Dictionary with debug information
        """
        # Execute action in base environment
        obs, base_reward, done, info = self.env.step(action)

        # Extract metrics from info
        distance = info.get('distance_to_cube', 0.0)
        alignment = info.get('alignment', 0.0)
        cube_visible = info.get('cube_visible', False)
        collision_count = 0  # Collisions disabled in C1.py

        # Track episode statistics
        self.episode_distances.append(distance)
        self.episode_alignments.append(alignment)
        self.episode_total_frames += 1
        if cube_visible:
            self.episode_visibility_frames += 1

        # Compute curriculum-based reward
        if self.enable_curriculum:
            current_stage = self.curriculum_manager.get_current_stage()
            reward_function = self.reward_functions[current_stage]

            # Call reward function with temporal data
            reward, reward_info = reward_function(
                distance, alignment, cube_visible, collision_count,
                prev_distance=self.prev_distance,
                visibility_history=self.visibility_history
            )

            # Update temporal tracking for next step
            self.prev_distance = distance
            self.visibility_history.append(cube_visible)

            # Add stage info to debug dict
            info['curriculum_stage'] = current_stage
            info['stage_config'] = self.curriculum_manager.get_stage_config()
            info['success_rate'] = self.curriculum_manager.get_success_rate()
            info['reward_components'] = reward_info
        else:
            # Use baseline C1.py reward
            reward = base_reward
            info['curriculum_stage'] = "BASELINE"

        # Return in old Gym API format for Stable-Baselines3 compatibility
        # (SB3 doesn't support new Gymnasium 5-tuple API yet)
        return obs, reward, done, info

    def get_curriculum_stats(self) -> Dict:
        """
        Get current curriculum statistics.

        Returns:
            Dictionary with stage, success rate, episode count
        """
        return {
            'stage': self.curriculum_manager.get_current_stage(),
            'success_rate': self.curriculum_manager.get_success_rate(),
            'episode_count': self.curriculum_manager.episode_count,
            'total_episodes': self.curriculum_manager.total_episodes,
            'stage_config': self.curriculum_manager.get_stage_config()
        }

    def force_advance_stage(self):
        """Manually advance to next stage (for testing/debugging)."""
        if self.curriculum_manager.current_stage < 5:
            self.curriculum_manager.advance_stage()


# Test the wrapper
if __name__ == "__main__":
    print("="*70)
    print("CURRICULUM ENVIRONMENT TEST")
    print("="*70)

    # Create curriculum environment
    env = CurriculumCubeTrackingEnv(log_dir="test_curriculum_env_logs")

    print(f"\nInitial observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\nStarting curriculum stage: {env.curriculum_manager.get_current_stage()}")

    # Run a few episodes
    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        print(f"\n{'─'*70}")
        print(f"Episode {episode + 1}/{num_episodes} | Stage {env.curriculum_manager.get_current_stage()}")
        print(f"{'─'*70}")

        done = False
        while not done:
            action = env.action_space.sample()  # Random action for testing
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Print every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count:3d} | Reward: {reward:+8.2f} | "
                      f"Distance: {info['distance_to_cube']:.4f} | "
                      f"Visible: {info['cube_visible']} | "
                      f"Stage: {info['curriculum_stage']}")

        print(f"\nEpisode Summary:")
        print(f"  Total Steps: {step_count}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Distance: {info['distance_to_cube']:.4f}")
        print(f"  Final Alignment: {info['alignment']:.4f}")

        # Show curriculum stats
        stats = env.get_curriculum_stats()
        print(f"\nCurriculum Stats:")
        print(f"  Current Stage: {stats['stage']}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Episodes in Stage: {stats['episode_count']}")
        print(f"  Total Episodes: {stats['total_episodes']}")

    env.close()
    print(f"\n{'='*70}")
    print("✓ Test complete! Check 'test_curriculum_env_logs/' for results")
    print(f"{'='*70}\n")
