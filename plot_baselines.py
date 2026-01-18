"""
Baseline Policy Visualization

Generates plots from baseline evaluation CSV logs to verify reward probe gates.
Shows oracle > scripted > random separation across all 6 curriculum stages.

Usage:
    # After running evaluate_baselines.py
    python plot_baselines.py --input baseline_eval

    # Custom output directory
    python plot_baselines.py --input baseline_eval --output plots
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_baseline_data(input_dir):
    """
    Load all baseline CSV logs from evaluation directory.

    Returns:
        dict: {policy_name: {stage: DataFrame}}
    """
    data = {
        'Random': {},
        'Scripted': {},
        'Oracle': {}
    }

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)

        # Parse filename: {policy}_stage{N}.csv
        if 'random' in filename.lower():
            policy = 'Random'
        elif 'scripted' in filename.lower():
            policy = 'Scripted'
        elif 'oracle' in filename.lower():
            policy = 'Oracle'
        else:
            continue

        # Extract stage number
        stage_str = filename.split('stage')[1].replace('.csv', '')
        stage = int(stage_str)

        # Load CSV
        df = pd.read_csv(csv_file)
        data[policy][stage] = df

    return data


def plot_return_distributions(data, output_dir):
    """
    Box plot showing return distributions for each policy across stages.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline Policy Returns - Reward Probe Gate', fontsize=16, fontweight='bold')

    stages = sorted(data['Random'].keys())
    policies = ['Random', 'Scripted', 'Oracle']
    colors = {'Random': '#e74c3c', 'Scripted': '#3498db', 'Oracle': '#2ecc71'}

    for idx, stage in enumerate(stages):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Collect returns for each policy
        returns_by_policy = []
        labels = []

        for policy in policies:
            if stage in data[policy]:
                returns = data[policy][stage]['return'].values
                returns_by_policy.append(returns)
                labels.append(policy)

        # Box plot
        bp = ax.boxplot(returns_by_policy, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        # Color boxes
        for patch, policy in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[policy])
            patch.set_alpha(0.6)

        # Formatting
        ax.set_title(f'Stage {stage}', fontweight='bold')
        ax.set_ylabel('Episode Return')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    output_path = os.path.join(output_dir, 'baseline_returns_boxplot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_success_rates(data, output_dir):
    """
    Bar chart showing success rates per policy per stage.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = sorted(data['Random'].keys())
    policies = ['Random', 'Scripted', 'Oracle']
    colors = {'Random': '#e74c3c', 'Scripted': '#3498db', 'Oracle': '#2ecc71'}

    # Compute success rates
    success_rates = {policy: [] for policy in policies}

    for stage in stages:
        for policy in policies:
            if stage in data[policy]:
                df = data[policy][stage]
                success_rate = df['success'].mean()
                success_rates[policy].append(success_rate * 100)
            else:
                success_rates[policy].append(0)

    # Bar positions
    x = np.arange(len(stages))
    width = 0.25

    # Plot bars
    for i, policy in enumerate(policies):
        offset = (i - 1) * width
        ax.bar(x + offset, success_rates[policy], width,
               label=policy, color=colors[policy], alpha=0.8)

    # Formatting
    ax.set_xlabel('Curriculum Stage', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Baseline Policy Success Rates Across Stages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Stage {s}' for s in stages])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'baseline_success_rates.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_median_comparison(data, output_dir):
    """
    Line plot showing median returns across stages for each policy.
    Verifies oracle > scripted > random separation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = sorted(data['Random'].keys())
    policies = ['Random', 'Scripted', 'Oracle']
    colors = {'Random': '#e74c3c', 'Scripted': '#3498db', 'Oracle': '#2ecc71'}
    markers = {'Random': 'o', 'Scripted': 's', 'Oracle': '^'}

    for policy in policies:
        medians = []
        q25 = []
        q75 = []

        for stage in stages:
            if stage in data[policy]:
                returns = data[policy][stage]['return'].values
                medians.append(np.median(returns))
                q25.append(np.percentile(returns, 25))
                q75.append(np.percentile(returns, 75))
            else:
                medians.append(0)
                q25.append(0)
                q75.append(0)

        # Plot median with error band
        ax.plot(stages, medians, marker=markers[policy], label=policy,
                color=colors[policy], linewidth=2, markersize=8)
        ax.fill_between(stages, q25, q75, color=colors[policy], alpha=0.2)

    # Formatting
    ax.set_xlabel('Curriculum Stage', fontweight='bold')
    ax.set_ylabel('Median Episode Return', fontweight='bold')
    ax.set_title('Reward Probe Gate: Policy Performance Separation', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(stages)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'baseline_median_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_visibility_and_distance(data, output_dir):
    """
    Scatter plots showing final distance and visibility rates.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stages = sorted(data['Random'].keys())
    policies = ['Random', 'Scripted', 'Oracle']
    colors = {'Random': '#e74c3c', 'Scripted': '#3498db', 'Oracle': '#2ecc71'}

    # Distance plot
    for policy in policies:
        mean_distances = []

        for stage in stages:
            if stage in data[policy]:
                distances = data[policy][stage]['distance_final'].values
                mean_distances.append(np.mean(distances))
            else:
                mean_distances.append(0)

        ax1.plot(stages, mean_distances, marker='o', label=policy,
                color=colors[policy], linewidth=2, markersize=8)

    ax1.set_xlabel('Curriculum Stage', fontweight='bold')
    ax1.set_ylabel('Mean Final Distance to Cube (m)', fontweight='bold')
    ax1.set_title('Final Distance to Cube', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stages)

    # Visibility plot
    for policy in policies:
        mean_visibility = []

        for stage in stages:
            if stage in data[policy]:
                visibility = data[policy][stage]['visibility_rate'].values
                mean_visibility.append(np.mean(visibility) * 100)
            else:
                mean_visibility.append(0)

        ax2.plot(stages, mean_visibility, marker='o', label=policy,
                color=colors[policy], linewidth=2, markersize=8)

    ax2.set_xlabel('Curriculum Stage', fontweight='bold')
    ax2.set_ylabel('Mean Visibility Rate (%)', fontweight='bold')
    ax2.set_title('Cube Visibility Rate', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(stages)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'baseline_distance_visibility.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_gate_report(data):
    """
    Print pass/fail report for reward probe gate.

    Gate passes if: median(oracle) > median(scripted) > median(random)
    """
    print("\n" + "="*70)
    print("REWARD PROBE GATE REPORT")
    print("="*70)

    stages = sorted(data['Random'].keys())

    all_passed = True

    for stage in stages:
        random_median = np.median(data['Random'][stage]['return'].values) if stage in data['Random'] else 0
        scripted_median = np.median(data['Scripted'][stage]['return'].values) if stage in data['Scripted'] else 0
        oracle_median = np.median(data['Oracle'][stage]['return'].values) if stage in data['Oracle'] else 0

        # Check gate condition
        oracle_gt_scripted = oracle_median > scripted_median
        scripted_gt_random = scripted_median > random_median

        passed = oracle_gt_scripted and scripted_gt_random
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"\nStage {stage}: {status}")
        print(f"  Random:   {random_median:>8.1f}")
        print(f"  Scripted: {scripted_median:>8.1f}")
        print(f"  Oracle:   {oracle_median:>8.1f}")

        if not oracle_gt_scripted:
            print(f"  ⚠️  Oracle not better than Scripted!")
        if not scripted_gt_random:
            print(f"  ⚠️  Scripted not better than Random!")

    print("\n" + "="*70)
    if all_passed:
        print("✅ OVERALL: ALL STAGES PASSED - Reward functions are learnable")
    else:
        print("❌ OVERALL: SOME STAGES FAILED - Review reward design")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize baseline policy evaluation results')
    parser.add_argument('--input', type=str, default='baseline_eval',
                       help='Input directory with CSV logs')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("Loading baseline evaluation data...")
    data = load_baseline_data(args.input)

    # Check if data exists
    if not any(data['Random']):
        print(f"❌ No data found in {args.input}/")
        print(f"   Run: python evaluate_baselines.py --mode headless --episodes 50")
        return

    print(f"Found data for {len(data['Random'])} stages")

    # Generate plots
    print("\nGenerating plots...")
    plot_return_distributions(data, args.output)
    plot_success_rates(data, args.output)
    plot_median_comparison(data, args.output)
    plot_visibility_and_distance(data, args.output)

    # Print gate report
    print_gate_report(data)

    print(f"\n✓ All plots saved to: {args.output}/")
    print(f"  - baseline_returns_boxplot.png")
    print(f"  - baseline_success_rates.png")
    print(f"  - baseline_median_comparison.png")
    print(f"  - baseline_distance_visibility.png")


if __name__ == "__main__":
    main()
