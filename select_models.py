"""
Model Selector - Find and List Trained Models
==============================================

Lists all available trained models with metadata.
Helps identify best models for demonstration and evaluation.
"""

import os
import glob
import re
from datetime import datetime


def parse_model_name(filename):
    """Extract metadata from model filename"""
    parts = filename.split('_')

    info = {
        'full_name': filename.replace('_actor.pth', ''),
        'type': 'unknown',
        'envs': 'unknown',
        'date': 'unknown',
        'tag': 'none'
    }

    # Extract model type
    if 'OPTIMIZED' in filename or 'VANILLA' in filename:
        info['type'] = 'Vanilla'
    elif 'ANTIHOPPING' in filename:
        info['type'] = 'Anti-Hopping'
    elif 'BALANCED' in filename:
        info['type'] = 'Balanced'
    elif 'PARALLEL' in filename:
        info['type'] = 'Parallel'

    # Extract number of parallel environments
    for part in parts:
        if 'envs' in part:
            info['envs'] = part

    # Extract training date
    date_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    match = re.search(date_pattern, filename)
    if match:
        info['date'] = match.group(1)

    # Extract model tag (BEST, FINAL, checkpoint)
    if 'BEST' in filename:
        info['tag'] = 'BEST'
    elif 'FINAL' in filename:
        info['tag'] = 'FINAL'
    elif 'step_' in filename:
        step_match = re.search(r'step_(\d+)', filename)
        if step_match:
            info['tag'] = f"step_{step_match.group(1)}"

    return info


def list_models(models_dir='models'):
    """List all available trained models"""

    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory not found: {models_dir}")
        return []

    # Find all actor model files
    actor_files = glob.glob(os.path.join(models_dir, '*_actor.pth'))

    if not actor_files:
        print(f"[WARNING] No models found in {models_dir}")
        return []

    models = []
    for actor_file in actor_files:
        basename = os.path.basename(actor_file)
        info = parse_model_name(basename)

        # Get file size
        size_mb = os.path.getsize(actor_file) / (1024 * 1024)
        info['size_mb'] = size_mb

        models.append(info)

    return models


def print_models_table(models):
    """Print models in formatted table"""

    print("\n" + "=" * 100)
    print("AVAILABLE MODELS")
    print("=" * 100)
    print(f"{'#':<4} {'Type':<15} {'Envs':<10} {'Tag':<15} {'Date':<20} {'Size (MB)':<10}")
    print("-" * 100)

    for i, model in enumerate(models, 1):
        print(f"{i:<4} {model['type']:<15} {model['envs']:<10} {model['tag']:<15} "
              f"{model['date']:<20} {model['size_mb']:.1f}")

    print("=" * 100 + "\n")


def suggest_models(models):
    """Suggest best models for demo"""

    print("\n" + "=" * 100)
    print("RECOMMENDED MODELS FOR DEMO")
    print("=" * 100 + "\n")

    # Get BEST models
    best_models = [m for m in models if m['tag'] == 'BEST']
    best_models_sorted = sorted(best_models, key=lambda x: x['date'], reverse=True)

    print("Strategy 1: Best Performance Comparison")
    print("-" * 100)
    print("Show your top performing models from different approaches\n")

    if len(best_models_sorted) >= 4:
        for i, model in enumerate(best_models_sorted[:4], 1):
            print(f"{i}. {model['type']:<15} | {model['envs']:<10} | {model['full_name']}")
    else:
        print(f"[WARNING] Found {len(best_models_sorted)} BEST models (need 4 for 2x2 grid)")
        print("\nAvailable BEST models:")
        for model in best_models_sorted:
            print(f"  - {model['type']:<15} | {model['full_name']}")

    print()

    # Learning progression
    print("\nStrategy 2: Learning Progression")
    print("-" * 100)
    print("Show learning progress with checkpoint models\n")

    step_models = [m for m in models if 'step_' in m['tag']]
    step_models_sorted = sorted(step_models, key=lambda x: int(x['tag'].split('_')[1]))

    print("Available checkpoint models:")
    for model in step_models_sorted:
        print(f"  - {model['tag']:<15} | {model['type']:<15} | {model['full_name']}")

    print()

    # Problem-solving story
    print("\nStrategy 3: Problem-Solution Story")
    print("-" * 100)
    print("Demonstrate problem identification and solution\n")

    vanilla = [m for m in best_models if 'Vanilla' in m['type']]
    antihopping = [m for m in best_models if 'Anti-Hopping' in m['type']]

    print("Recommended:")
    if vanilla:
        print(f"  Baseline:     {vanilla[0]['full_name']}")
    if antihopping:
        print(f"  Anti-Hopping: {antihopping[0]['full_name']}")

    print("\n" + "=" * 100 + "\n")


def main():
    """Main function"""

    print("\n" + "=" * 100)
    print("MODEL SELECTOR")
    print("=" * 100)

    # List all models
    models = list_models('models')

    if not models:
        print("[ERROR] No models found. Train some models first.")
        return

    # Print table
    print_models_table(models)

    # Suggest best models
    suggest_models(models)

    # Instructions
    print("=" * 100)
    print("USAGE")
    print("=" * 100)
    print()
    print("1. Choose 4 models from the table above")
    print("2. Copy their full names (without '_actor.pth')")
    print("3. Edit demo_2x2.py model_configs:")
    print()
    print("   model_configs = [")
    print("       {'name': 'Model A', 'path': 'FULL_NAME_HERE'},")
    print("       {'name': 'Model B', 'path': 'FULL_NAME_HERE'},")
    print("       {'name': 'Model C', 'path': 'FULL_NAME_HERE'},")
    print("       {'name': 'Model D', 'path': 'FULL_NAME_HERE'},")
    print("   ]")
    print()
    print("4. Run: python demo_2x2.py")
    print()
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()