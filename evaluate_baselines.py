"""
Baseline Policy Evaluation Harness

Runs all 3 baseline policies (Random, Scripted, Oracle) across all 6 curriculum stages.
Generates CSV logs and summary statistics for reward probe gates.

Usage:
    # Full evaluation (3 policies × 6 stages × 50 episodes = 900 episodes)
    python evaluate_baselines.py --mode headless --episodes 50

    # Quick test (fewer episodes)
    python evaluate_baselines.py --mode headless --episodes 10 --stages 1 2 3

    # Visual verification in GUI
    python evaluate_baselines.py --mode gui --episodes 5 --stages 3
"""

import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict

from baseline_policies import RandomPolicy, ScriptedPolicy, OraclePolicy
from curriculum_env import CurriculumCubeTrackingEnv
from utils.episode_logger import EpisodeLogger

# Force unbuffered output
sys.stdout = sys.stderr


def evaluate_policy(policy, policy_name, env, stage, num_episodes, logger):
    """
    Evaluate a policy on a specific stage.

    Returns:
        dict: Summary statistics
    """
    results = {
        'returns': [],
        'distances': [],
        'alignments': [],
        'visibility_ratios': [],
        'successes': []
    }

    print(f"\n[Stage {stage}] Evaluating {policy_name} Policy ({num_episodes} episodes)...")
    start_time = time.time()

    for ep in range(num_episodes):
        ep_start = time.time()
        obs = env.reset()
        done = False
        episode_return = 0
        step_count = 0
        visibility_count = 0
        max_steps_per_episode = 300  # Match env max_steps

        # Reset policy state
        if hasattr(policy, 'reset'):
            policy.reset()

        while not done and step_count < max_steps_per_episode:
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

            episode_return += reward
            step_count += 1

            if info.get('cube_visible', False):
                visibility_count += 1

        ep_time = time.time() - ep_start
        print(f"    Ep {ep+1}: {step_count} steps in {ep_time:.1f}s, dist={info['distance_to_cube']:.3f}m")

        # Record metrics
        results['returns'].append(episode_return)
        results['distances'].append(info['distance_to_cube'])
        results['alignments'].append(info['alignment'])
        visibility_rate = visibility_count / max(step_count, 1)
        results['visibility_ratios'].append(visibility_rate)

        # Check success based on stage criteria
        success = info.get('success', False)
        results['successes'].append(success)

        # Log to CSV
        logger.log_episode(
            episode=ep,
            stage=stage,
            episode_return=episode_return,
            distance_final=info['distance_to_cube'],
            alignment_final=info['alignment'],
            visibility_rate=visibility_rate,
            collisions=info.get('collisions', 0),
            termination_reason=info.get('termination_reason', 'timeout'),
            success=success
        )

        # Progress indicator
        if (ep + 1) % 10 == 0:
            success_rate = sum(results['successes']) / len(results['successes'])
            mean_return = np.mean(results['returns'])
            print(f"  Episodes {ep+1}/{num_episodes}: "
                  f"Success={success_rate:.1%}, "
                  f"MeanReturn={mean_return:.1f}")

    elapsed = time.time() - start_time

    # Summary
    summary = {
        'success_rate': np.mean(results['successes']),
        'mean_return': np.mean(results['returns']),
        'median_return': np.median(results['returns']),
        'std_return': np.std(results['returns']),
        'mean_distance': np.mean(results['distances']),
        'mean_alignment': np.mean(results['alignments']),
        'mean_visibility_ratio': np.mean(results['visibility_ratios']),
        'time_elapsed': elapsed
    }

    print(f"  ✓ Complete in {elapsed:.1f}s | "
          f"Success: {summary['success_rate']:.1%} | "
          f"Median Return: {summary['median_return']:.1f}")

    return summary


def run_full_evaluation(mode='headless', episodes_per_config=50, stages=None, output_dir='baseline_eval'):
    """
    Run full baseline evaluation: 3 policies × N stages × M episodes.

    Args:
        mode: 'gui' or 'headless'
        episodes_per_config: Episodes per (policy, stage) pair
        stages: List of stages to evaluate (default: [1,2,3,4,5,6])
        output_dir: Directory for logs and results
    """
    if stages is None:
        stages = [1, 2, 3, 4, 5, 6]

    print("="*70)
    print("BASELINE POLICY EVALUATION - REWARD PROBE GATES")
    print("="*70)
    print(f"Mode: {mode}")
    print(f"Episodes per (policy, stage): {episodes_per_config}")
    print(f"Stages: {stages}")
    print(f"Total episodes: {3 * len(stages) * episodes_per_config}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    results = defaultdict(dict)
    policy_names = ['Random', 'Scripted', 'Oracle']

    for stage in stages:
        print(f"\n{'='*70}")
        print(f"EVALUATING STAGE {stage}")
        print(f"{'='*70}")

        # Create environment for this stage
        env = CurriculumCubeTrackingEnv(
            log_dir=f"{output_dir}/stage{stage}",
            enable_curriculum=False,  # Fix stage
            mode=mode
        )
        env.curriculum_manager.current_stage = stage

        # Random Policy
        log_path = f"{output_dir}/random_stage{stage}.csv"
        with EpisodeLogger(log_path) as logger:
            random_policy = RandomPolicy(env.action_space)
            results[stage]['Random'] = evaluate_policy(
                random_policy, 'Random', env, stage, episodes_per_config, logger
            )

        # Scripted Policy
        log_path = f"{output_dir}/scripted_stage{stage}.csv"
        with EpisodeLogger(log_path) as logger:
            scripted_policy = ScriptedPolicy(
                env.env, env.env.robot_id, env.env.end_effector_index, env.env.cube_id, stage
            )
            results[stage]['Scripted'] = evaluate_policy(
                scripted_policy, 'Scripted', env, stage, episodes_per_config, logger
            )

        # Oracle Policy (uses privileged state from PyBullet)
        log_path = f"{output_dir}/oracle_stage{stage}.csv"
        with EpisodeLogger(log_path) as logger:
            oracle_policy = OraclePolicy(
                env.env, env.env.robot_id, env.env.end_effector_index,
                env.env.cube_id,  # Privileged access to true cube position
                stochastic=True
            )
            results[stage]['Oracle'] = evaluate_policy(
                oracle_policy, 'Oracle', env, stage, episodes_per_config, logger
            )

        env.close()

    # Print summary table
    print("\n" + "="*70)
    print("BASELINE EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Stage':<8} {'Policy':<12} {'Success %':<12} {'Median Return':<15} {'Visibility %'}")
    print("-"*70)

    for stage in stages:
        for policy_name in policy_names:
            res = results[stage][policy_name]
            print(f"{stage:<8} {policy_name:<12} {res['success_rate']*100:>10.1f}% "
                  f"{res['median_return']:>14.1f} {res['mean_visibility_ratio']*100:>12.1f}%")

    print("="*70)

    # Save summary to file
    summary_path = f"{output_dir}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write("BASELINE EVALUATION SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"{'Stage':<8} {'Policy':<12} {'Success %':<12} {'Median Return':<15} {'Visibility %'}\n")
        f.write("-"*70 + "\n")

        for stage in stages:
            for policy_name in policy_names:
                res = results[stage][policy_name]
                f.write(f"{stage:<8} {policy_name:<12} {res['success_rate']*100:>10.1f}% "
                       f"{res['median_return']:>14.1f} {res['mean_visibility_ratio']*100:>12.1f}%\n")

    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"✓ CSV logs saved to: {output_dir}/{{policy}}_stage{{N}}.csv")
    print(f"\nNext step: python plot_baselines.py --input {output_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate baseline policies across curriculum stages')
    parser.add_argument('--mode', type=str, default='headless',
                       choices=['gui', 'headless'],
                       help='GUI for visual verification, headless for speed')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes per (policy, stage) pair')
    parser.add_argument('--stages', type=int, nargs='+', default=None,
                       help='Stages to evaluate (default: all 1-6)')
    parser.add_argument('--output', type=str, default='baseline_eval',
                       help='Output directory for logs and results')

    args = parser.parse_args()

    results = run_full_evaluation(
        mode=args.mode,
        episodes_per_config=args.episodes,
        stages=args.stages,
        output_dir=args.output
    )
