"""
Episode Logger - CSV metrics tracking for RL experiments

Logs per-episode metrics to CSV for:
- Quick inspection in Excel/Google Sheets
- Plotting success rates, returns, distances over time
- Detecting plateaus and convergence

Usage:
    logger = EpisodeLogger('logs/experiment_1.csv')

    for episode in range(1000):
        obs = env.reset()
        done = False
        episode_return = 0
        visibility_count = 0
        steps = 0

        while not done:
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            visibility_count += int(info.get('cube_visible', False))
            steps += 1

        logger.log_episode(
            episode=episode,
            stage=env.current_stage,
            episode_return=episode_return,
            distance_final=info['distance_to_cube'],
            alignment_final=info['alignment'],
            visibility_rate=visibility_count / steps,
            collisions=info.get('collisions', 0),
            termination_reason=info.get('termination_reason', 'unknown'),
            success=info.get('success', False)
        )
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional


class EpisodeLogger:
    """
    Logs episode-level metrics to CSV for analysis.

    Features:
    - Append-only (safe for resuming experiments)
    - Creates parent directories automatically
    - Flushes after each write (safe for crashes)
    - Includes timestamp for each episode
    """

    def __init__(self, log_path: str):
        """
        Initialize episode logger.

        Args:
            log_path: Path to CSV file (will be created if doesn't exist)
        """
        self.log_path = log_path
        self.fieldnames = [
            'timestamp',
            'episode',
            'stage',
            'return',
            'distance_final',
            'alignment_final',
            'visibility_rate',
            'collisions',
            'termination_reason',
            'success'
        ]

        # Create parent directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Check if file exists
        file_exists = os.path.exists(log_path)

        # Open file in append mode
        self.file = open(log_path, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        # Write header if new file
        if not file_exists or os.path.getsize(log_path) == 0:
            self.writer.writeheader()
            self.file.flush()

        print(f"Episode logger initialized: {log_path}")
        print(f"Logging fields: {', '.join(self.fieldnames)}")

    def log_episode(self,
                   episode: int,
                   stage: int,
                   episode_return: float,
                   distance_final: float,
                   alignment_final: float,
                   visibility_rate: float,
                   collisions: int = 0,
                   termination_reason: str = 'unknown',
                   success: bool = False):
        """
        Log metrics for a single episode.

        Args:
            episode: Episode number
            stage: Curriculum stage (1-6)
            episode_return: Cumulative reward for episode
            distance_final: Final distance to cube (meters)
            alignment_final: Final alignment score
            visibility_rate: Fraction of steps with cube visible (0-1)
            collisions: Number of collision events
            termination_reason: 'success', 'timeout', 'collision', 'lost_object', etc.
            success: Whether episode succeeded (met stage criteria)
        """
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'episode': episode,
            'stage': stage,
            'return': f"{episode_return:.2f}",
            'distance_final': f"{distance_final:.4f}",
            'alignment_final': f"{alignment_final:.3f}",
            'visibility_rate': f"{visibility_rate:.3f}",
            'collisions': collisions,
            'termination_reason': termination_reason,
            'success': int(success)
        }

        self.writer.writerow(row)
        self.file.flush()  # Flush immediately (crash-safe)

    def close(self):
        """Close the log file."""
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
            print(f"Episode logger closed: {self.log_path}")

    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()


def load_episode_log(log_path: str) -> Dict[str, list]:
    """
    Load episode log CSV into dictionary of lists.

    Args:
        log_path: Path to CSV file

    Returns:
        Dictionary with keys matching fieldnames, values as lists

    Example:
        data = load_episode_log('logs/experiment.csv')
        episodes = data['episode']
        returns = data['return']

        import matplotlib.pyplot as plt
        plt.plot(episodes, returns)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show()
    """
    data = {
        'timestamp': [],
        'episode': [],
        'stage': [],
        'return': [],
        'distance_final': [],
        'alignment_final': [],
        'visibility_rate': [],
        'collisions': [],
        'termination_reason': [],
        'success': []
    }

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return data

    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['timestamp'].append(row['timestamp'])
            data['episode'].append(int(row['episode']))
            data['stage'].append(int(row['stage']))
            data['return'].append(float(row['return']))
            data['distance_final'].append(float(row['distance_final']))
            data['alignment_final'].append(float(row['alignment_final']))
            data['visibility_rate'].append(float(row['visibility_rate']))
            data['collisions'].append(int(row['collisions']))
            data['termination_reason'].append(row['termination_reason'])
            data['success'].append(bool(int(row['success'])))

    print(f"Loaded {len(data['episode'])} episodes from {log_path}")
    return data


def compute_summary_stats(log_path: str, window_size: int = 100) -> Dict[str, Any]:
    """
    Compute summary statistics from episode log.

    Args:
        log_path: Path to CSV file
        window_size: Rolling window size for success rate

    Returns:
        Dictionary with summary statistics
    """
    data = load_episode_log(log_path)

    if len(data['episode']) == 0:
        return {}

    # Overall stats
    total_episodes = len(data['episode'])
    overall_success_rate = sum(data['success']) / total_episodes if total_episodes > 0 else 0
    mean_return = sum(data['return']) / total_episodes if total_episodes > 0 else 0
    mean_distance = sum(data['distance_final']) / total_episodes if total_episodes > 0 else 0

    # Recent window stats (last N episodes)
    recent_episodes = min(window_size, total_episodes)
    recent_success_rate = sum(data['success'][-recent_episodes:]) / recent_episodes if recent_episodes > 0 else 0
    recent_mean_return = sum(data['return'][-recent_episodes:]) / recent_episodes if recent_episodes > 0 else 0

    # Per-stage stats
    stages = set(data['stage'])
    stage_stats = {}
    for stage in sorted(stages):
        stage_episodes = [i for i, s in enumerate(data['stage']) if s == stage]
        if stage_episodes:
            stage_success = [data['success'][i] for i in stage_episodes]
            stage_stats[stage] = {
                'episodes': len(stage_episodes),
                'success_rate': sum(stage_success) / len(stage_success),
                'mean_return': sum([data['return'][i] for i in stage_episodes]) / len(stage_episodes)
            }

    return {
        'total_episodes': total_episodes,
        'overall_success_rate': overall_success_rate,
        'mean_return': mean_return,
        'mean_distance': mean_distance,
        'recent_success_rate': recent_success_rate,
        'recent_mean_return': recent_mean_return,
        'stage_stats': stage_stats
    }


if __name__ == "__main__":
    # Demo usage
    import numpy as np

    # Create demo log
    log_path = 'demo_episode_log.csv'

    with EpisodeLogger(log_path) as logger:
        # Simulate 100 episodes with improving performance
        for ep in range(100):
            stage = min(ep // 20 + 1, 6)

            # Simulate improving performance
            success_prob = min(ep / 100, 0.8)
            success = np.random.random() < success_prob

            episode_return = -1000 + ep * 10 + np.random.randn() * 100
            distance_final = 0.5 - ep * 0.004 + np.random.randn() * 0.05
            distance_final = max(0.01, distance_final)
            alignment_final = min(ep * 0.008, 0.9) + np.random.randn() * 0.1
            visibility_rate = min(0.5 + ep * 0.005, 0.95)

            logger.log_episode(
                episode=ep,
                stage=stage,
                episode_return=episode_return,
                distance_final=distance_final,
                alignment_final=alignment_final,
                visibility_rate=visibility_rate,
                collisions=0,
                termination_reason='success' if success else 'timeout',
                success=success
            )

    # Load and summarize
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    stats = compute_summary_stats(log_path, window_size=20)

    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Overall success rate: {stats['overall_success_rate']:.1%}")
    print(f"Mean return: {stats['mean_return']:.1f}")
    print(f"Recent success rate (last 20): {stats['recent_success_rate']:.1%}")
    print(f"Recent mean return (last 20): {stats['recent_mean_return']:.1f}")

    print("\nPer-stage statistics:")
    for stage, stage_stat in stats['stage_stats'].items():
        print(f"  Stage {stage}: {stage_stat['episodes']} episodes, "
              f"{stage_stat['success_rate']:.1%} success, "
              f"{stage_stat['mean_return']:.1f} mean return")

    print(f"\nDemo log saved to: {log_path}")
    print("You can open it in Excel/Google Sheets or use load_episode_log() to plot")
