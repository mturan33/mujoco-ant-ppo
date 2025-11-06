"""
PPO Agent Testing Script
========================

Tests a trained PPO agent in the Ant-v5 environment with visualization.
Evaluates agent performance over multiple episodes.
"""

import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import time
import numpy as np


def test():
    """Test trained PPO agent"""

    # Configuration
    model_name = "Ant-v5_PPO_ANTIHOPPING_16envs_2025-11-05_18-09-09_BEST"
    num_episodes = 10
    render_speed = 0.02  # seconds between frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")

    # Create environment with visualization
    env = gym.make('Ant-v5', render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[INFO] State Dimension: {state_dim}")
    print(f"[INFO] Action Dimension: {action_dim}")
    print(f"[INFO] Max Action: {max_action}")

    # Create agent
    agent = PPOAgent(
        state_dim, action_dim, max_action,
        actor_lr=3e-4, critic_lr=3e-4,
        gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, device=device
    )
    agent.to(device)

    # Load trained model
    try:
        agent.load("models", model_name)
        print(f"[OK] Loaded model: {model_name}")
    except FileNotFoundError:
        print(f"[ERROR] Model not found: {model_name}")
        print("[INFO] Available models in 'models' directory:")
        import os
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith('_actor.pth')]
            for f in files:
                print(f"  - {f.replace('_actor.pth', '')}")
        return

    print(f"\n{'=' * 60}")
    print(f"[START] Testing for {num_episodes} episodes")
    print(f"{'=' * 60}\n")

    episode_rewards = []

    # Run episodes
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            with torch.no_grad():
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

                # Normalize using training statistics
                norm_state_tensor = torch.clamp(
                    (state_tensor - agent.obs_rms.mean) / torch.sqrt(agent.obs_rms.var + 1e-8),
                    -10.0, 10.0
                )

                # Get deterministic action (mean, not sample)
                action_dist = agent.actor(norm_state_tensor)
                action = action_dist.mean

            # Execute action
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            # Control render speed
            time.sleep(render_speed)

        episode_rewards.append(total_reward)
        print(f"[EPISODE {episode_idx + 1}/{num_episodes}] "
              f"Reward: {total_reward:.2f} | Steps: {step_count}")

    env.close()

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"[RESULTS] Testing Complete")
    print(f"{'=' * 60}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    test()