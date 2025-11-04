"""
FINAL MAIN.PY - STABLE WALKING ANT (NO HOPPING!)
=================================================

ZIPLAMA PROBLEMİ FİX EDİLDİ! ✅

ÖNCEKİ SORUN:
- Agent "reward hacking" yapıyordu
- Zıplayarak hem forward velocity hem height bonus alıyordu
- Energy/jerk penalty'ler çok zayıftı

YENİ DEĞİŞİKLİKLER:
1. ✅ Stability-Aware Height Bonus
   - Sadece stabil yükseklikte ise bonus (zıplama = 0 bonus!)

2. ✅ Stability-Aware Forward Bonus
   - Stabil ilerleme: 1.2x bonus
   - Zıplama: 0.3x bonus (caydırıcı!)

3. ✅ Aggressive Penalties
   - Energy: 0.01 → 0.05 (5x daha güçlü!)
   - Jerk: 0.05 → 0.2 (4x daha güçlü!)

4. ✅ Total Timesteps: 4M
   - İlk 1M: Eski behavior'dan kurtulma
   - 2-3M: Yeni reward'a adapte
   - 3-4M: Convergence

5. ✅ Target Reward: 1500 (gerçekçi)

HEDEF: Düzgün, stabil, energy-efficient yürüme! 🚀
"""

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
    # ========================================
    # HİPERPARAMETRELER - OPTİMİZE EDİLDİ
    # ========================================

    total_timesteps = 10_000_000  # FİNAL: 4M (zıplama fix + convergence)

    # Learning rates
    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4  # YENİ: 1e-3 → 3e-4 (critic loss çok yüksekti)

    # PPO parametreleri
    gamma = 0.99
    gae_lambda = 0.95
    rollout_steps = 2048
    update_epochs = 10
    clip_ratio = 0.2
    batch_size = 64
    entropy_coef = 0.01  # YENİ: 0.02 → 0.01 (daha az exploration)

    # Checkpoint ayarları
    save_freq = 500_000
    eval_freq = 100_000
    target_reward = 1500  # GÜNCEL: Yeni reward shaping ile gerçekçi hedef

    # ========================================
    # YENİ: ENERGY EFFICIENCY AYARLARI
    # ========================================

    # Action magnitude penalty (enerji tasarrufu)
    action_penalty_coef = 0.05  # ARTTIRILDI: 0.01 → 0.05 (zıplama önleme!)

    # Action smoothness penalty (pürüzsüz hareket)
    jerk_penalty_coef = 0.2  # ARTTIRILDI: 0.05 → 0.2 (zıplama önleme!)

    # Deney ayarları
    load_model = False
    experiment_name = "Ant-v5_PPO_FINAL_STABLE"  # FİNAL: Zıplama fix!
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

    # Ortamı yükle
    env = gym.make('Ant-v5')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    log_to_file(f"[INFO] State Dimension: {state_dim}")
    log_to_file(f"[INFO] Action Dimension: {action_dim}")
    log_to_file(f"[INFO] Max Action Value: {max_action}")
    log_to_file(f"[INFO] Total Timesteps: {total_timesteps:,}")
    log_to_file(
        f"[INFO] Energy Efficiency: ON (action_penalty={action_penalty_coef}, jerk_penalty={jerk_penalty_coef})")

    # Agent oluştur
    agent = PPOAgent(
        state_dim, action_dim, max_action,
        actor_learning_rate, critic_learning_rate,
        gamma, gae_lambda, clip_ratio, device
    )
    agent.to(device)

    # Learning rate scheduler
    def lr_lambda(step):
        return max(0.5, 1.0 - (step / total_timesteps))  # Min 0.5x

    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda)

    if load_model:
        try:
            agent.load("models", experiment_name)
            log_to_file("[OK] Pre-trained model loaded.")
        except FileNotFoundError:
            log_to_file("[WARNING] Model not found, starting from scratch.")

    # Ortamı sıfırla
    state, info = env.reset()
    global_step_count = 0
    current_episode_reward = 0
    episode_rewards_list = []
    episode_count = 0
    best_avg_reward = -np.inf

    # ========================================
    # YENİ: PREVIOUS ACTION TRACKING
    # ========================================
    previous_action = np.zeros(action_dim)  # İlk action 0

    log_to_file(f"\n{'=' * 60}\n[START] ENERGY-EFFICIENT TRAINING STARTED\n{'=' * 60}\n")

    # Ana eğitim döngüsü
    while global_step_count < total_timesteps:
        # --- ROLLOUT (Veri Toplama) ---
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(rollout_steps):
            global_step_count += 1

            # Action seç (yeni get_action metodu)
            action, log_prob, value = agent.get_action(state.reshape(1, -1))

            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated

            # ========================================
            # YENİ: STABILITY-AWARE REWARD SHAPING (ZIPLAMA FİX!)
            # ========================================

            # 1. Orijinal reward
            base_reward = reward

            # 2. Stability check (yükseklik değişimi)
            torso_height = next_state[0]
            previous_height = state[0]
            height_change = abs(torso_height - previous_height)

            # Stability threshold: 0.05 (zıplarken 0.1+ değişir!)
            is_stable = height_change < 0.05

            # 3. Height bonus (sadece stabil + yüksek ise)
            if is_stable and torso_height > 0.3:
                height_bonus = 1.5  # Stabil yürüme ÖDÜL!
            elif torso_height > 0.3:
                height_bonus = 0.0  # Yüksek ama stabil değil (zıplıyor)
            else:
                height_bonus = -2.0  # Düşük = kötü

            # 4. Forward velocity bonus (sadece stabil hareket)
            forward_velocity = info.get('x_velocity', 0)
            if is_stable:
                forward_bonus = forward_velocity * 1.2  # Stabil ilerleme ÖDÜL!
            else:
                forward_bonus = forward_velocity * 0.3  # Zıplama = az bonus

            # 5. Energy efficiency penalty
            action_np = action.cpu().numpy().flatten()
            action_magnitude = np.sum(np.square(action_np))
            energy_penalty = -action_penalty_coef * action_magnitude

            # 6. Action smoothness penalty (jerk)
            action_diff = np.abs(action_np - previous_action)
            jerk_magnitude = np.sum(action_diff)
            jerk_penalty = -jerk_penalty_coef * jerk_magnitude

            # Toplam reward (shaped)
            shaped_reward = (
                    base_reward +
                    height_bonus +
                    forward_bonus +
                    energy_penalty +
                    jerk_penalty
            )

            # Previous action güncelle
            previous_action = action_np.copy()

            # Reward'ı normalize et
            normalized_reward = agent.normalize_reward(shaped_reward)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(normalized_reward)
            dones.append(done)
            values.append(value)

            state = next_state
            current_episode_reward += base_reward  # Log original reward

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

                # TensorBoard logging
                writer.add_scalar('episode/reward', current_episode_reward, global_step_count)
                writer.add_scalar('episode/avg_reward_100', avg_reward, global_step_count)

                current_episode_reward = 0
                state, info = env.reset()
                previous_action = np.zeros(action_dim)  # Reset previous action

        # --- ADVANTAGE COMPUTATION ---
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            # Normalize state for value prediction
            norm_next_state = torch.clamp(
                (next_state_tensor - agent.obs_rms.mean) / torch.sqrt(agent.obs_rms.var + 1e-8),
                -10.0, 10.0
            )
            next_value = agent.critic(norm_next_state)

        advantages, returns = agent.compute_advantages(rewards, dones, values, next_value)

        # --- LEARNING ---
        actor_loss, critic_loss, avg_std = agent.learn(
            states, actions, log_probs, returns, advantages,
            update_epochs, batch_size, entropy_coef
        )

        actor_scheduler.step()
        critic_scheduler.step()

        # --- TENSORBOARD LOGGING ---
        writer.add_scalar('losses/actor_loss', actor_loss, global_step_count)
        writer.add_scalar('losses/critic_loss', critic_loss, global_step_count)
        writer.add_scalar('charts/exploration_std', avg_std, global_step_count)
        writer.add_scalar('charts/learning_rate', actor_scheduler.get_last_lr()[0], global_step_count)

        if len(episode_rewards_list) > 0:
            writer.add_scalar('charts/avg_reward_100_episodes',
                              np.mean(episode_rewards_list[-100:]), global_step_count)

        # --- CHECKPOINT KAYDETME ---
        if global_step_count % save_freq == 0:
            checkpoint_name = f"{run_name}_step_{global_step_count}"
            agent.save("models", checkpoint_name)
            log_to_file(f"[CHECKPOINT] Saved: {checkpoint_name}")

        # --- BEST MODEL KAYDETME ---
        if len(episode_rewards_list) >= 100:
            current_avg = np.mean(episode_rewards_list[-100:])
            if current_avg > best_avg_reward:
                best_avg_reward = current_avg
                agent.save("models", f"{run_name}_BEST")
                log_to_file(f"[BEST MODEL] New record! Avg Reward: {best_avg_reward:.2f}")

        # --- EARLY SUCCESS CHECK ---
        if len(episode_rewards_list) >= 100:
            if np.mean(episode_rewards_list[-100:]) >= target_reward:
                log_to_file(f"\n[SUCCESS] Target reward {target_reward} achieved!")
                log_to_file(f"Final Avg Reward: {np.mean(episode_rewards_list[-100:]):.2f}")
                break

        # --- PROGRESS REPORT ---
        if global_step_count % eval_freq == 0:
            progress = (global_step_count / total_timesteps) * 100
            if len(episode_rewards_list) >= 100:
                recent_avg = np.mean(episode_rewards_list[-100:])
                log_to_file(
                    f"\n[PROGRESS REPORT]\n"
                    f"   Steps: {global_step_count:,} / {total_timesteps:,} ({progress:.1f}%)\n"
                    f"   Avg Reward (last 100): {recent_avg:.2f}\n"
                    f"   Actor Loss: {actor_loss:.4f}\n"
                    f"   Critic Loss: {critic_loss:.4f}\n"
                    f"   Exploration Std: {avg_std:.4f}\n"
                    f"   Learning Rate: {actor_scheduler.get_last_lr()[0]:.6f}\n"
                )

    # Final save
    agent.save("models", f"{run_name}_FINAL")
    log_to_file(f"\n{'=' * 60}")
    log_to_file(f"[COMPLETE] ENERGY-EFFICIENT TRAINING FINISHED!")
    log_to_file(f"{'=' * 60}")

    if len(episode_rewards_list) >= 100:
        log_to_file(f"Final Avg Reward: {np.mean(episode_rewards_list[-100:]):.2f}")

    writer.close()
    env.close()


if __name__ == '__main__':
    main()