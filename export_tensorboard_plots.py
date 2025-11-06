"""
TensorBoard Plot Exporter
==========================

Exports TensorBoard metrics as high-quality PNG images for documentation.
Generates publication-ready plots with consistent styling.

Usage:
    python export_tensorboard_plots.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path


def export_tensorboard_plots(log_dir='./runs', output_dir='./plots', models_to_compare=None):
    """
    Export TensorBoard logs as PNG plots

    Args:
        log_dir: Directory containing TensorBoard event files
        output_dir: Output directory for PNG images
        models_to_compare: List of model name patterns to filter (None = export all)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print(f"TENSORBOARD PLOT EXPORTER")
    print(f"{'='*60}\n")

    # Find all run directories
    run_dirs = [d for d in Path(log_dir).iterdir() if d.is_dir()]

    if not run_dirs:
        print(f"[ERROR] No runs found in {log_dir}")
        return

    print(f"[INFO] Found {len(run_dirs)} runs")

    # Filter runs if specified
    if models_to_compare:
        run_dirs = [d for d in run_dirs if any(model in d.name for model in models_to_compare)]
        print(f"[INFO] Filtered to {len(run_dirs)} runs matching: {models_to_compare}")

    # Metrics to extract
    metrics_to_plot = [
        'episode/reward',
        'episode/avg_reward_100',
        'losses/actor_loss',
        'losses/critic_loss',
        'charts/exploration_std',
        'charts/learning_rate',
        'performance/steps_per_second'
    ]

    # Extract data from event files
    all_data = {metric: {} for metric in metrics_to_plot}

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"[LOADING] {run_name}...")

        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()

            available_tags = ea.Tags()['scalars']

            for metric in metrics_to_plot:
                if metric in available_tags:
                    events = ea.Scalars(metric)
                    data = [(e.step, e.value) for e in events]
                    all_data[metric][run_name] = data

        except Exception as e:
            print(f"[WARNING] Could not load {run_name}: {e}")

    # Generate plots
    _plot_training_rewards(all_data, output_dir)
    _plot_loss_curves(all_data, output_dir)
    _plot_exploration(all_data, output_dir)
    _plot_final_comparison(all_data, output_dir)
    _plot_training_speed(all_data, output_dir)

    print(f"\n{'='*60}")
    print(f"[COMPLETE] All plots saved to: {output_dir}/")
    print(f"{'='*60}\n")

    print("\nGENERATED PLOTS:")
    print("1. training_rewards.png    - Learning curve")
    print("2. loss_curves.png         - Actor & Critic losses")
    print("3. exploration_std.png     - Exploration decay")
    print("4. final_comparison.png    - Performance comparison")
    print("5. training_speed.png      - Training efficiency")


def _plot_training_rewards(all_data, output_dir):
    """Plot training reward curves"""
    if 'episode/avg_reward_100' in all_data and all_data['episode/avg_reward_100']:
        plt.figure(figsize=(16, 9))

        for run_name, data in all_data['episode/avg_reward_100'].items():
            if data:
                steps, rewards = zip(*data)
                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')
                plt.plot(steps, rewards, label=clean_name, linewidth=2, alpha=0.8)

        plt.xlabel('Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Reward (100 episodes)', fontsize=14, fontweight='bold')
        plt.title('PPO Training Progress - Ant-v5 Locomotion', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'training_rewards.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()


def _plot_loss_curves(all_data, output_dir):
    """Plot actor and critic loss curves"""
    if all_data['losses/actor_loss'] or all_data['losses/critic_loss']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Actor loss
        for run_name, data in all_data['losses/actor_loss'].items():
            if data:
                steps, losses = zip(*data)
                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')
                ax1.plot(steps, losses, label=clean_name, linewidth=2, alpha=0.8)

        ax1.set_xlabel('Steps', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actor Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Actor Network Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Critic loss
        for run_name, data in all_data['losses/critic_loss'].items():
            if data:
                steps, losses = zip(*data)
                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')
                ax2.plot(steps, losses, label=clean_name, linewidth=2, alpha=0.8)

        ax2.set_xlabel('Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Critic Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Critic Network Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()


def _plot_exploration(all_data, output_dir):
    """Plot exploration (action std) decay"""
    if 'charts/exploration_std' in all_data and all_data['charts/exploration_std']:
        plt.figure(figsize=(12, 6))

        for run_name, data in all_data['charts/exploration_std'].items():
            if data:
                steps, stds = zip(*data)
                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')
                plt.plot(steps, stds, label=clean_name, linewidth=2, alpha=0.8)

        plt.xlabel('Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Standard Deviation', fontsize=14, fontweight='bold')
        plt.title('Exploration Decay (Action Std)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'exploration_std.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()


def _plot_final_comparison(all_data, output_dir):
    """Plot final performance comparison (bar chart)"""
    if 'episode/avg_reward_100' in all_data:
        final_rewards = {}

        for run_name, data in all_data['episode/avg_reward_100'].items():
            if data:
                # Average of last 10% of training
                num_points = len(data)
                last_10_percent = data[int(num_points * 0.9):]
                avg_reward = np.mean([v for _, v in last_10_percent])

                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')[:30]
                final_rewards[clean_name] = avg_reward

        if final_rewards:
            plt.figure(figsize=(12, 7))

            names = list(final_rewards.keys())
            rewards = list(final_rewards.values())

            # Color by performance
            colors = ['green' if r > 2000 else 'orange' if r > 1000 else 'red' for r in rewards]

            bars = plt.bar(names, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

            # Add value labels
            for bar, reward in zip(bars, rewards):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{reward:.0f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11)

            plt.xlabel('Model', fontsize=14, fontweight='bold')
            plt.ylabel('Final Average Reward', fontsize=14, fontweight='bold')
            plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(output_dir, 'final_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] {output_path}")
            plt.close()


def _plot_training_speed(all_data, output_dir):
    """Plot training speed (steps per second)"""
    if 'performance/steps_per_second' in all_data and all_data['performance/steps_per_second']:
        plt.figure(figsize=(12, 6))

        for run_name, data in all_data['performance/steps_per_second'].items():
            if data:
                steps, speeds = zip(*data)
                clean_name = run_name.replace('Ant-v5_PPO_', '').replace('_2025-', ' ')
                plt.plot(steps, speeds, label=clean_name, linewidth=2, alpha=0.8)

        plt.xlabel('Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Steps/Second', fontsize=14, fontweight='bold')
        plt.title('Training Speed Performance', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'training_speed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()


def main():
    """Main function with configuration"""

    print("\n" + "="*60)
    print("TENSORBOARD EXPORT OPTIONS")
    print("="*60)
    print("\n1. Export ALL runs")
    print("2. Export specific models (recommended for comparison)")
    print("\nChoice: ", end='')

    choice = input().strip()

    if choice == '1':
        export_tensorboard_plots()
    else:
        # Export selected models for comparison
        top_models = [
            'OPTIMIZED_12envs',
            'ANTIHOPPING_16envs',
            'MINIMAL',
            'ENERGY_EFFICIENT'
        ]

        print(f"\n[INFO] Exporting comparison of selected models...")
        export_tensorboard_plots(models_to_compare=top_models)


if __name__ == '__main__':
    main()