"""
Curriculum Manager for 5-stage progressive reinforcement learning.
Tracks episode success rates and automatically advances through stages.

Based on user's C1.py design:
- Stage 1: Visibility only (±10 reward)
- Stage 2: Visibility + Distance (-distance² ± 10)
- Stage 3: Add partial alignment (alignment × 5)
- Stage 4: Full alignment (alignment × 10)
- Stage 5: Sparse rewards for final policy

Uses ground-truth PyBullet data (not DOPE) for reliable training.
"""

from collections import deque
from typing import Dict
import csv
import json
from datetime import datetime
import os
import numpy as np


class CurriculumManager:
    """
    Manages curriculum learning with 5 progressive stages.

    Stages build on user's C1.py implementation:
    1. Visibility only (70% threshold)
    2. Visibility + Distance - USER'S CURRENT C1.py (60%)
    3. Visibility + Distance + Partial Alignment (50%)
    4. Visibility + Distance + Full Alignment (40%)
    5. Sparse rewards for publishable results (35%)
    """

    def __init__(self, log_dir: str = "curriculum_logs"):
        """
        Initialize curriculum manager.

        Args:
            log_dir: Directory to save stage transition logs
        """
        self.current_stage = 1
        self.episode_count = 0
        self.total_episodes = 0  # Track across all stages
        self.success_window = deque(maxlen=100)  # Rolling window for success rate

        # Initialize tracking attributes for logging
        self.last_distance = None
        self.last_alignment = None

        # Adaptive episodes: fewer for easy stages, more for hard stages
        self.min_episodes_per_stage_map = {
            1: 300,   # Visibility only
            2: 350,   # NEW: Visibility + Easy Distance (temporal tolerance)
            3: 400,   # Visibility + Medium Distance
            4: 500,   # Partial alignment
            5: 600,   # Full alignment - harder, needs more practice
            6: None   # Train until total_timesteps reached
        }

        # Stage-specific success thresholds
        self.stage_thresholds = {
            1: 0.70,  # 70% success on visibility task
            2: 0.65,  # 65% success on visibility + easy distance
            3: 0.60,  # 60% success on visibility + medium distance
            4: 0.50,  # 50% success with partial alignment
            5: 0.40,  # 40% success with full alignment
            6: 0.35   # 35% success on complete sparse task
        }

        # Success criteria per stage
        # Smoother curriculum with more gradual distance progression
        self.stage_criteria = {
            1: {"visibility": True},                                      # Just see the cube
            2: {"visibility": True, "distance": 0.35},                    # Easy distance (relaxed from 0.25)
            3: {"visibility": True, "distance": 0.25},                    # Medium distance
            4: {"visibility": True, "distance": 0.20},                    # Tighter distance
            5: {"visibility": True, "distance": 0.15, "alignment": 0.3}, # Distance + partial alignment
            6: {"visibility": True, "distance": 0.12, "alignment": 0.5}  # Final precision
        }

        # Setup logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "stage_transitions.csv")
        self._init_log_file()

    def _init_log_file(self):
        """Initialize CSV log file with headers."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'episode', 'total_episodes',
                'success_rate', 'avg_distance', 'avg_alignment', 'transition'
            ])

    def reset(self):
        """Reset manager to stage 1."""
        self.current_stage = 1
        self.episode_count = 0
        self.success_window.clear()

    def check_episode_success(self, info_dict: Dict) -> bool:
        """
        Check if episode meets current stage success criteria.

        Args:
            info_dict: Dictionary with 'distance_to_cube', 'alignment', 'cube_visible'

        Returns:
            True if episode successful for current stage
        """
        criteria = self.stage_criteria[self.current_stage]

        # Check visibility (all stages require this)
        if not info_dict.get('cube_visible', False):
            return False

        # Check distance if required
        if 'distance' in criteria:
            if info_dict.get('distance_to_cube', float('inf')) >= criteria['distance']:
                return False

        # Check alignment if required
        if 'alignment' in criteria:
            if info_dict.get('alignment', -1) <= criteria['alignment']:
                return False

        return True

    def record_episode(self, success: bool, distance: float = None, alignment: float = None):
        """
        Record an episode outcome and update tracking.

        Args:
            success: Whether the episode was successful based on stage criteria
            distance: Average distance to cube during episode (optional, for logging)
            alignment: Average alignment during episode (optional, for logging)
        """
        self.episode_count += 1
        self.total_episodes += 1
        self.success_window.append(1 if success else 0)

        # Store for logging
        self.last_distance = distance
        self.last_alignment = alignment

    def get_success_rate(self) -> float:
        """
        Get current success rate from rolling window.

        Returns:
            Success rate (0.0 to 1.0) or 0.0 if no episodes recorded
        """
        if len(self.success_window) == 0:
            return 0.0
        return sum(self.success_window) / len(self.success_window)

    def should_advance_stage(self) -> bool:
        """
        Check if curriculum should advance to next stage.

        Returns:
            True if ready to advance, False otherwise
        """
        # Don't advance if already at final stage
        if self.current_stage >= 6:
            return False

        # Use stage-specific minimum episodes
        min_episodes = self.min_episodes_per_stage_map.get(self.current_stage, 500)
        if min_episodes is not None and self.episode_count < min_episodes:
            return False

        # Check if success rate meets threshold
        success_rate = self.get_success_rate()
        threshold = self.stage_thresholds[self.current_stage]

        return success_rate >= threshold

    def advance_stage(self, model=None):
        """
        Advance to next curriculum stage and reset tracking.

        Args:
            model: SAC model reference for buffer clearing (optional but recommended)
        """
        if self.current_stage < 6:
            # Log transition
            self._log_transition(advancing=True)

            # Advance stage
            self.current_stage += 1
            self.episode_count = 0
            self.success_window.clear()

            # CRITICAL: Clear replay buffer to remove stale reward data
            # Old experiences have rewards from previous stage's reward function
            if model is not None and hasattr(model, 'replay_buffer'):
                buffer_size = model.replay_buffer.buffer_size
                old_pos = model.replay_buffer.pos
                model.replay_buffer.reset()
                print(f"✓ Cleared replay buffer ({old_pos} experiences removed, capacity: {buffer_size})")

            print(f"\n{'='*70}")
            print(f"CURRICULUM ADVANCED TO STAGE {self.current_stage}")
            print(f"{'='*70}")
            print(f"New success criteria: {self.stage_criteria[self.current_stage]}")
            print(f"Target success rate: {self.stage_thresholds[self.current_stage]*100:.0f}%")
            print(f"{'='*70}\n")

    def get_current_stage(self) -> int:
        """Get current curriculum stage (1-5)."""
        return self.current_stage

    def get_stage_config(self) -> Dict[str, any]:
        """
        Get configuration for current stage.

        Returns:
            Dictionary with stage parameters:
            - stage: Current stage number
            - use_distance_reward: Whether to include distance in reward
            - use_alignment_reward: Whether to include alignment in reward
            - alignment_weight: Weight for alignment reward (0, 5, or 10)
            - use_visibility_reward: Whether to include visibility in reward
            - sparse_reward: Whether to use sparse (vs dense) rewards
            - distance_weight: Weight for distance penalty
        """
        configs = {
            1: {
                'use_distance': False,
                'use_alignment': False,
                'alignment_weight': 0,
                'use_visibility': True,
                'sparse': False,
                'distance_weight': 0
            },
            2: {
                'use_distance': True,
                'use_alignment': False,
                'alignment_weight': 0,
                'use_visibility': True,
                'sparse': False,
                'distance_weight': 1.0,  # Full -distance²
                'temporal_tolerance': True  # NEW: Enable blind approach tolerance
            },
            3: {
                'use_distance': True,
                'use_alignment': False,
                'alignment_weight': 0,
                'use_visibility': True,
                'sparse': False,
                'distance_weight': 1.0  # Full -distance²
            },
            4: {
                'use_distance': True,
                'use_alignment': True,
                'alignment_weight': 5,  # Partial alignment
                'use_visibility': True,
                'sparse': False,
                'distance_weight': 1.0
            },
            5: {
                'use_distance': True,
                'use_alignment': True,
                'alignment_weight': 10,  # Full alignment (user's original)
                'use_visibility': True,
                'sparse': False,
                'distance_weight': 1.0
            },
            6: {
                'use_distance': True,
                'use_alignment': True,
                'alignment_weight': 10,
                'use_visibility': True,
                'sparse': True,  # Sparse rewards
                'distance_weight': 0.2  # Minimal shaping
            }
        }

        config = configs[self.current_stage]
        config['stage'] = self.current_stage
        config['success_criteria'] = self.stage_criteria[self.current_stage]
        return config

    def _log_transition(self, advancing: bool = False):
        """Log stage transition to CSV."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success_rate = self.get_success_rate()

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.current_stage,
                self.episode_count,
                self.total_episodes,
                f"{success_rate:.3f}",
                f"{self.last_distance:.3f}" if self.last_distance else "N/A",
                f"{self.last_alignment:.3f}" if self.last_alignment else "N/A",
                'ADVANCE' if advancing else 'UPDATE'
            ])

    def periodic_log(self, force: bool = False):
        """
        Periodically log progress (every 100 episodes or when forced).

        Args:
            force: Force logging regardless of episode count
        """
        if force or (self.episode_count % 100 == 0 and self.episode_count > 0):
            self._log_transition(advancing=False)
            self.save_state()  # Persist state for crash recovery

            # Console update
            success_rate = self.get_success_rate()
            threshold = self.stage_thresholds[self.current_stage]
            progress = (success_rate / threshold) * 100 if threshold > 0 else 0
            min_episodes = self.min_episodes_per_stage_map.get(self.current_stage, 500)
            min_episodes_display = "N/A" if min_episodes is None else str(min_episodes)

            print(f"Stage {self.current_stage} | Episode {self.episode_count}/{min_episodes_display} | "
                  f"Success: {success_rate:.2%} (target: {threshold:.0%}) | "
                  f"Progress: {progress:.0f}%")

    def save_state(self, filepath: str = None):
        """Save curriculum state to JSON for crash recovery."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "curriculum_state.json")

        state = {
            'current_stage': self.current_stage,
            'episode_count': self.episode_count,
            'total_episodes': self.total_episodes,
            'success_window': list(self.success_window),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str = None) -> bool:
        """Load curriculum state from JSON. Returns True if successful."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "curriculum_state.json")

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Validate required keys exist
            required = ['current_stage', 'episode_count', 'total_episodes', 'success_window']
            for key in required:
                if key not in state:
                    raise ValueError(f"Missing required key: {key}")

            # Validate stage range
            if not isinstance(state['current_stage'], int) or not (1 <= state['current_stage'] <= 6):
                raise ValueError(f"Invalid stage: {state['current_stage']}")

            # Validate success_window is a list
            if not isinstance(state['success_window'], list):
                raise ValueError(f"success_window must be list, got {type(state['success_window'])}")

            self.current_stage = state['current_stage']
            self.episode_count = state['episode_count']
            self.total_episodes = state['total_episodes']
            self.success_window = deque(state['success_window'], maxlen=100)
            print(f"[OK] Loaded curriculum state: Stage {self.current_stage}")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load curriculum state: {e}")
            return False


# Example usage for testing
if __name__ == "__main__":
    print("="*70)
    print("CURRICULUM MANAGER TEST")
    print("="*70)

    manager = CurriculumManager(log_dir="test_curriculum_logs")

    print(f"\nStarting at stage {manager.get_current_stage()}")
    print(f"Stage config: {manager.get_stage_config()}")

    # Simulate some episodes
    print("\nSimulating training episodes...\n")

    for episode in range(700):
        # Simulate increasing success rate
        success_prob = 0.5 + (episode / 1000)  # Gradually improve
        success = (np.random.random() < success_prob)

        # Simulate metrics
        distance = 0.15 - (episode / 10000)  # Gradually get closer
        alignment = 0.3 + (episode / 2000)   # Gradually improve alignment

        manager.record_episode(success, distance, alignment)

        # Check for stage advancement
        if manager.should_advance_stage():
            print(f"\n✨ After {manager.episode_count} episodes, success rate: {manager.get_success_rate():.2%}")
            manager.advance_stage()
            print(f"New config: {manager.get_stage_config()}\n")

        # Periodic logging
        if episode % 100 == 0 and episode > 0:
            manager.periodic_log()

    print(f"\n{'='*70}")
    print(f"Final stage reached: {manager.get_current_stage()}")
    print(f"Total episodes across all stages: {manager.total_episodes}")
    print(f"{'='*70}")
    print(f"\n✓ Test complete! Check 'test_curriculum_logs/stage_transitions.csv' for results")
