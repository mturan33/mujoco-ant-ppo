# test_agent.py
import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import time


def test():
    # --- Ayarlar ---
    model_name = "Ant-v5_PPO_10M"  # Yüklemek istediğiniz modelin adı
    num_episodes = 10  # Kaç bölüm test etmek istediğiniz

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ortamı 'human' modunda oluşturarak görselleştirmeyi sağlıyoruz
    env = gym.make('Ant-v5', render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Agent'ı oluştur (hiperparametreler önemli değil, sadece ağ yapısı için gerekli)
    agent = PPOAgent(state_dim, action_dim, max_action, 0, 0, 0, 0)
    agent.to(device)

    try:
        agent.load("models", model_name)
    except FileNotFoundError:
        print(f"HATA: '{model_name}' adlı model 'models' klasöründe bulunamadı.")
        return

    for i in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

            with torch.no_grad():
                action_dist = agent.actor(state_tensor)
                action = action_dist.mean  # Test sırasında en olası eylemi seçiyoruz (deterministik)

            state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            total_reward += reward

            # Simülasyonu yavaşlatmak için (isteğe bağlı)
            # time.sleep(0.01)

        print(f"Bölüm {i + 1} tamamlandı. Toplam Ödül: {total_reward:.2f}")

    env.close()


if __name__ == '__main__':
    test()