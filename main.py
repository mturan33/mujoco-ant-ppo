import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")


def main():
    # MINIMAL APPROACH - Original reward'a güven!
    total_timesteps = 15_000_000
    actor_learning_rate = 3e-4
    critic_learning_rate = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    rollout_steps = 2048
    update_epochs = 10
    clip_ratio = 0.2
    batch_size = 64
    entropy_coef = 0.01

    save_freq = 500_000
    eval_freq = 100_000
    target_reward = 6000

    load_model = False
    experiment_name = "Ant-v5_PPO_MINIMAL"
    run_name = f"{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./runs"):
        os.makedirs("./runs")

    log_file_path = f"./runs/{run_name}.txt"
    writer = SummaryWriter(f"./runs/{run_name}")

    def log_to_file(message):
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)

    env = gym.make('Ant-v5')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    log_to_file(f"[INFO] State Dimension: {state_dim}")
    log_to_file(f"[INFO] Action Dimension: {action_dim}")
    log_to_file(f"[INFO] Max Action Value: {max_action}")
    log_to_file(f"[INFO] Total Timesteps: {total_timesteps:,}")
    log_to_file(f"[WARNING] MINIMAL MODE - Original reward + small healthy bonus!")

    agent = PPOAgent(
        state_dim, action_dim, max_action,
        actor_learning_rate, critic_learning_rate,
        gamma, gae_lambda, clip_ratio, device
    )
    agent.to(device)

    def lr_lambda(step):
        return max(0.5, 1.0 - (step / total_timesteps))

    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda)

    if load_model:
        try:
            agent.load("models", experiment_name)
            log_to_file("[OK] Pre-trained model loaded.")
        except FileNotFoundError:
            log_to_file("[WARNING] Model not found, starting from scratch.")

    state, info = env.reset()
    global_step_count = 0
    current_episode_reward = 0
    episode_rewards_list = []
    episode_count = 0
    best_avg_reward = -np.inf

    log_to_file(f"\n{'=' * 60}\n[START] MINIMAL TRAINING STARTED\n{'=' * 60}\n")

    while global_step_count < total_timesteps:
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(rollout_steps):
            global_step_count += 1
            action, log_prob, value = agent.get_action(state.reshape(1, -1))

            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated

            # MINIMAL REWARD SHAPING - Original reward'a güven!
            # Sadece küçük bir healthy bonus ekle
            torso_height = next_state[0]  # İlk element torso z-position
            healthy_bonus = 0.5 if torso_height > 0.3 else 0.0

            # Original reward + küçük healthy bonus
            shaped_reward = reward + healthy_bonus

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            normalized_reward = agent.normalize_reward(shaped_reward)
            rewards.append(normalized_reward)
            dones.append(done)
            values.append(value)

            state = next_state
            current_episode_reward += reward

            if done:
                episode_count += 1
                episode_rewards_list.append(current_episode_reward)
                avg_reward = np.mean(episode_rewards_list[-100:])

                log_message = (
                    f"[EPISODE {episode_count}] "
                    f"Step: {global_step_count:,} | "
                    f"Reward: {current_episode_reward:.2f} | "
                    f"Avg-100: {avg_reward:.2f}"
                )
                log_to_file(log_message)

                writer.add_scalar('episode/reward', current_episode_reward, global_step_count)
                writer.add_scalar('episode/avg_reward_100', avg_reward, global_step_count)

                current_episode_reward = 0
                state, info = env.reset()

        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            next_value = agent.critic(next_state_tensor)

        advantages, returns = agent.compute_advantages(rewards, dones, values, next_value)

        actor_loss, critic_loss, avg_std = agent.learn(
            states, actions, log_probs, returns, advantages,
            update_epochs, batch_size, entropy_coef
        )

        actor_scheduler.step()
        critic_scheduler.step()

        writer.add_scalar('losses/actor_loss', actor_loss, global_step_count)
        writer.add_scalar('losses/critic_loss', critic_loss, global_step_count)
        writer.add_scalar('charts/exploration_std', avg_std, global_step_count)
        writer.add_scalar('charts/learning_rate', actor_scheduler.get_last_lr()[0], global_step_count)

        if len(episode_rewards_list) > 0:
            writer.add_scalar('charts/avg_reward_100_episodes',
                              np.mean(episode_rewards_list[-100:]), global_step_count)

        if global_step_count % save_freq == 0:
            checkpoint_name = f"{run_name}_step_{global_step_count}"
            agent.save("models", checkpoint_name)
            log_to_file(f"[CHECKPOINT] Saved: {checkpoint_name}")

        if len(episode_rewards_list) >= 100:
            current_avg = np.mean(episode_rewards_list[-100:])
            if current_avg > best_avg_reward:
                best_avg_reward = current_avg
                agent.save("models", f"{run_name}_BEST")
                log_to_file(f"[BEST MODEL] New record! Avg Reward: {best_avg_reward:.2f}")

        if len(episode_rewards_list) >= 100:
            if np.mean(episode_rewards_list[-100:]) >= target_reward:
                log_to_file(f"\n[SUCCESS] Target reward {target_reward} achieved!")
                log_to_file(f"Final Avg Reward: {np.mean(episode_rewards_list[-100:]):.2f}")
                break

        if global_step_count % eval_freq == 0:
            progress = (global_step_count / total_timesteps) * 100
            if len(episode_rewards_list) >= 100:
                recent_avg = np.mean(episode_rewards_list[-100:])
                log_to_file(
                    f"\n[MINIMAL REPORT]\n"
                    f"   Steps: {global_step_count:,} / {total_timesteps:,} ({progress:.1f}%)\n"
                    f"   Avg Reward (last 100): {recent_avg:.2f}\n"
                    f"   Actor Loss: {actor_loss:.4f}\n"
                    f"   Critic Loss: {critic_loss:.4f}\n"
                    f"   Exploration Std: {avg_std:.4f}\n"
                    f"   Learning Rate: {actor_scheduler.get_last_lr()[0]:.6f}\n"
                )

    agent.save("models", f"{run_name}_FINAL")
    log_to_file(f"\n{'=' * 60}")
    log_to_file(f"[COMPLETE] MINIMAL TRAINING FINISHED!")
    log_to_file(f"{'=' * 60}")

    if len(episode_rewards_list) >= 100:
        log_to_file(f"Final Avg Reward: {np.mean(episode_rewards_list[-100:]):.2f}")

    writer.close()
    env.close()


if __name__ == '__main__':
    main()