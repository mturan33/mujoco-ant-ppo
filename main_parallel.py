"""
ANTI-HOPPING ANT - HEIGHT STABILITY PENALTY! 🚀
================================================

SORUN BULUNDU:
- Vanilla Ant-v5 reward yeterli değilmiş!
- contact_cost = 5e-4 (çok zayıf!)
- Agent zıplamayı öğreniyor (daha hızlı = daha fazla reward)

ÇÖZÜM: HEIGHT STABILITY PENALTY (Minimal & Effective!)
- Sadece 1 penalty: Height variance
- Zıplama = Height çok değişir = Yüksek penalty!
- Düzgün yürüme = Height stabil = Düşük penalty!

```python
height_variance = abs(current_height - previous_height)
penalty = -5.0 * height_variance

# Zıplama: variance=0.3 → penalty=-1.5
# Normal: variance=0.02 → penalty=-0.1
```

NEDEN BU ÇALIŞIR?
- Basit (1 penalty!)
- Direkt zıplamayı hedefler
- Base reward'ı boğmaz
- Agent kafası karışmaz

BEKLENEN:
- Süre: ~5-6 saat (aynı)
- Reward: +2000-2500 (biraz düşük, ama stabil!)
- YÜRÜME: Pürüzsüz, zıplama yok! ✅
"""

import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv  # SubprocVecEnv yerine!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")


def make_env(rank):
    """Environment factory"""
    def _init():
        env = gym.make('Ant-v5')
        env.reset(seed=42 + rank)
        return env
    return _init


def main():
    # ========================================
    # PARAMETERS (OPTIMIZED!)
    # ========================================

    total_timesteps = 8_000_000
    num_envs = 16

    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    rollout_steps = 512  # Aynı (2048 çok uzun)
    update_epochs = 10
    clip_ratio = 0.2
    batch_size = 64
    entropy_coef = 0.01

    save_freq = 500_000
    eval_freq = 100_000

    load_model = False
    experiment_name = f"Ant-v5_PPO_ANTIHOPPING_{num_envs}envs"
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

    # ========================================
    # PARALLEL ENVIRONMENTS (DummyVecEnv!)
    # ========================================

    log_to_file(f"[INFO] Creating {num_envs} parallel environments...")
    log_to_file(f"[INFO] Using DummyVecEnv (no IPC overhead!)")

    # DummyVecEnv: Single process, sequential
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    temp_env = gym.make('Ant-v5')
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    max_action = float(temp_env.action_space.high[0])
    temp_env.close()

    log_to_file(f"[INFO] State Dimension: {state_dim}")
    log_to_file(f"[INFO] Action Dimension: {action_dim}")
    log_to_file(f"[INFO] Parallel Envs: {num_envs}")
    log_to_file(f"[INFO] Rollout Steps: {rollout_steps}")
    log_to_file(f"[INFO] Expected Time: ~30-35 minutes for 8M steps!")
    log_to_file(f"[INFO] HEIGHT STABILITY PENALTY MODE!")
    log_to_file(f"[INFO] Penalty Coefficient: 5.0 (strong anti-hopping!)")

    # ========================================
    # AGENT (REWARD NORMALIZATION KAPALI!)
    # ========================================

    agent = PPOAgent(
        state_dim, action_dim, max_action,
        actor_learning_rate, critic_learning_rate,
        gamma, gae_lambda, clip_ratio, device
    )
    agent.to(device)

    # Reward normalization KAPAT! (hızlanma)
    agent.normalize_rewards = False

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

    # ========================================
    # TRAINING LOOP
    # ========================================

    states = env.reset()
    global_step_count = 0
    episode_rewards = np.zeros(num_envs)
    episode_rewards_list = []
    episode_count = 0
    best_avg_reward = -np.inf

    # Height stability tracking
    previous_heights = np.array([states[i][0] for i in range(num_envs)])  # Height = state[0]
    height_stability_coef = 8.0  # Strong penalty!

    log_to_file(f"\n{'=' * 60}\n[START] ANTI-HOPPING TRAINING (Height Stability Penalty!)\n{'=' * 60}\n")

    import time
    start_time = time.time()
    last_log_time = start_time

    while global_step_count < total_timesteps:
        # ========================================
        # ROLLOUT (VANILLA - NO REWARD SHAPING!)
        # ========================================

        env_states = [[] for _ in range(num_envs)]
        env_actions = [[] for _ in range(num_envs)]
        env_log_probs = [[] for _ in range(num_envs)]
        env_rewards = [[] for _ in range(num_envs)]
        env_dones = [[] for _ in range(num_envs)]
        env_values = [[] for _ in range(num_envs)]

        for step in range(rollout_steps):
            global_step_count += num_envs

            with torch.no_grad():
                actions_list, log_probs_list, values_list = [], [], []
                for i in range(num_envs):
                    action, log_prob, value = agent.get_action(states[i].reshape(1, -1))
                    actions_list.append(action)
                    log_probs_list.append(log_prob)
                    values_list.append(value)

                actions = torch.cat(actions_list, dim=0)
                log_probs = torch.cat(log_probs_list, dim=0)
                values = torch.cat(values_list, dim=0)

            actions_np = actions.cpu().numpy()
            next_states, base_rewards, dones, infos = env.step(actions_np)

            # ========================================
            # HEIGHT STABILITY PENALTY!
            # ========================================

            # Base reward from Ant-v5
            shaped_rewards = base_rewards.copy()

            # Add height stability penalty
            for i in range(num_envs):
                current_height = next_states[i][0]  # Height = state[0]
                height_variance = abs(current_height - previous_heights[i])

                # Penalty: Zıplama = Yüksek variance = Büyük ceza!
                height_penalty = -height_stability_coef * height_variance

                shaped_rewards[i] += height_penalty

                # Update previous height
                if not dones[i]:  # Reset only on done
                    previous_heights[i] = current_height
                else:
                    previous_heights[i] = next_states[i][0]  # Reset to new episode start

            # Episode tracking
            for i in range(num_envs):
                episode_rewards[i] += base_rewards[i]

                if dones[i]:
                    episode_rewards_list.append(episode_rewards[i])
                    episode_count += 1

                    if len(episode_rewards_list) >= 100:
                        avg_reward = np.mean(episode_rewards_list[-100:])

                        # Hız hesaplama
                        current_time = time.time()
                        elapsed = current_time - last_log_time
                        steps_per_sec = eval_freq / elapsed if elapsed > 0 else 0

                        log_to_file(
                            f"[EPISODE {episode_count}] "
                            f"Step: {global_step_count:,} | "
                            f"Reward: {episode_rewards[i]:.2f} | "
                            f"Avg-100: {avg_reward:.2f} | "
                            f"Speed: {steps_per_sec:.0f} steps/s"
                        )
                        writer.add_scalar('episode/reward', episode_rewards[i], global_step_count)
                        writer.add_scalar('episode/avg_reward_100', avg_reward, global_step_count)
                        writer.add_scalar('performance/steps_per_second', steps_per_sec, global_step_count)

                        last_log_time = current_time

                    episode_rewards[i] = 0

            # Store data (ENV-MAJOR ORDER!)
            for i in range(num_envs):
                env_states[i].append(states[i])
                env_actions[i].append(actions_list[i])
                env_log_probs[i].append(log_probs_list[i])
                env_rewards[i].append(shaped_rewards[i])  # Use shaped rewards!
                env_dones[i].append(dones[i])
                env_values[i].append(values_list[i])

            states = next_states

        # ========================================
        # ADVANTAGE COMPUTATION
        # ========================================

        with torch.no_grad():
            next_states_tensor = torch.FloatTensor(states).to(device)
            next_values = []
            for i in range(num_envs):
                norm_state = torch.clamp(
                    (next_states_tensor[i:i+1] - agent.obs_rms.mean) /
                    torch.sqrt(agent.obs_rms.var + 1e-8),
                    -10.0, 10.0
                )
                next_value = agent.critic(norm_state)
                next_values.append(next_value)
            next_values = torch.cat(next_values, dim=0)

        all_states, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []

        for i in range(num_envs):
            advantages, returns = agent.compute_advantages(
                env_rewards[i], env_dones[i], env_values[i], next_values[i:i+1]
            )

            all_states.extend(env_states[i])
            all_actions.extend(env_actions[i])
            all_log_probs.extend(env_log_probs[i])
            all_advantages.append(advantages)
            all_returns.append(returns)

        flat_states = np.array(all_states)
        flat_actions = torch.cat(all_actions, dim=0)
        flat_log_probs = torch.cat(all_log_probs, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)

        # ========================================
        # LEARNING
        # ========================================

        actor_loss, critic_loss, avg_std = agent.learn(
            flat_states, flat_actions, flat_log_probs, returns, advantages,
            update_epochs, batch_size, entropy_coef
        )

        actor_scheduler.step()
        critic_scheduler.step()

        # ========================================
        # LOGGING & CHECKPOINTS
        # ========================================

        writer.add_scalar('losses/actor_loss', actor_loss, global_step_count)
        writer.add_scalar('losses/critic_loss', critic_loss, global_step_count)
        writer.add_scalar('charts/exploration_std', avg_std, global_step_count)
        writer.add_scalar('charts/learning_rate',
                         actor_scheduler.get_last_lr()[0], global_step_count)

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

        if global_step_count % eval_freq == 0:
            progress = (global_step_count / total_timesteps) * 100
            elapsed_total = time.time() - start_time
            remaining = (elapsed_total / global_step_count) * (total_timesteps - global_step_count)

            if len(episode_rewards_list) >= 100:
                recent_avg = np.mean(episode_rewards_list[-100:])
                log_to_file(
                    f"\n[PROGRESS REPORT]\n"
                    f"   Steps: {global_step_count:,} / {total_timesteps:,} ({progress:.1f}%)\n"
                    f"   Avg Reward (last 100): {recent_avg:.2f}\n"
                    f"   Elapsed: {elapsed_total/60:.1f} min\n"
                    f"   Remaining: {remaining/60:.1f} min\n"
                    f"   Actor Loss: {actor_loss:.4f}\n"
                    f"   Critic Loss: {critic_loss:.4f}\n"
                    f"   Exploration Std: {avg_std:.4f}\n"
                )

    # ========================================
    # FINISH
    # ========================================

    total_time = time.time() - start_time
    agent.save("models", f"{run_name}_FINAL")
    log_to_file(f"\n[COMPLETE] ANTI-HOPPING TRAINING FINISHED!")
    log_to_file(f"Total Time: {total_time/60:.1f} minutes")

    if len(episode_rewards_list) >= 100:
        log_to_file(f"Final Avg Reward: {np.mean(episode_rewards_list[-100:]):.2f}")

    writer.close()
    env.close()


if __name__ == '__main__':
    main()