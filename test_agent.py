import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import time
import numpy as np


def test():
    # --- Ayarlar ---
    # En son eğittiğiniz modelin adını buraya yazın
    model_name = "Ant-v5_PPO_MINIMAL_2025-11-03_13-20-59_BEST"  # BEST model'i kullan
    num_episodes = 10  # Kaç bölüm test edilecek
    render_speed = 0.02  # Render hızı (saniye)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")

    # Ortamı 'human' modunda oluştur
    env = gym.make('Ant-v5', render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"[INFO] State Dimension: {state_dim}")
    print(f"[INFO] Action Dimension: {action_dim}")
    print(f"[INFO] Max Action: {max_action}")

    # Agent'ı oluştur
    agent = PPOAgent(
        state_dim, action_dim, max_action,
        actor_lr=3e-4, critic_lr=1e-3,
        gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, device=device
    )
    agent.to(device)

    # Model yükle
    try:
        agent.load("models", model_name)
        print(f"[OK] '{model_name}' modeli başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"[ERROR] '{model_name}' bulunamadı!")
        print("[INFO] 'models' klasöründeki dosyalar:")
        import os
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith('_actor.pth')]
            for f in files:
                print(f"  - {f.replace('_actor.pth', '')}")
        return

    print(f"\n{'='*60}")
    print(f"[START] {num_episodes} EPISODE TEST BAŞLIYOR")
    print(f"{'='*60}\n")

    episode_rewards = []

    for episode_idx in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            with torch.no_grad():
                # 1. State'i tensor'e çevir
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

                # 2. Normalize et (eğitimde kullanılan istatistiklerle)
                norm_state_tensor = torch.clamp(
                    (state_tensor - agent.obs_rms.mean) / torch.sqrt(agent.obs_rms.var + 1e-8),
                    -10.0, 10.0
                )

                # 3. Deterministik eylem seç (mean kullan, sample değil)
                action_dist = agent.actor(norm_state_tensor)
                action = action_dist.mean  # Testte deterministik

            # Eylemi uygula
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            # Render'ı yavaşlat
            time.sleep(render_speed)
            # time.sleep(0.1)

        episode_rewards.append(total_reward)
        print(f"[EPISODE {episode_idx + 1}/{num_episodes}] "
              f"Reward: {total_reward:.2f} | Steps: {step_count}")

    env.close()

    # İstatistikler
    print(f"\n{'='*60}")
    print(f"[RESULTS] TEST TAMAMLANDI")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Dev: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    test()