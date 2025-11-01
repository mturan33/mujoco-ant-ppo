# test_agent.py
import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import time


def test():
    # --- Ayarlar ---
    model_name = "Ant-v5_PPO_T_2025-10-31_23-01-27"  # Yüklemek istediğiniz modelin adı
    num_episodes = 10  # Kaç bölüm test etmek istediğiniz

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan Cihaz: {device}")

    # Ortamı 'human' modunda oluşturarak görselleştirmeyi sağlıyoruz
    env = gym.make('Ant-v5', render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Agent'ı oluştur (hiperparametreler önemli değil, sadece ağ yapısı ve 'device' için gerekli)
    agent = PPOAgent(state_dim, action_dim, max_action, 0, 0, 0, 0, device)
    agent.to(device)

    try:
        agent.load("models", model_name)
        print(f"'{model_name}' modeli başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"HATA: '{model_name}' adlı model 'models' klasöründe bulunamadı.")
        return

    for i in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                # 1. Durumu tensöre çevir ve GPU'ya gönder
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

                # 2. Eğitimde kullanılan istatistiklerle durumu normalize et
                norm_state_tensor = torch.clamp(
                    (state_tensor - agent.obs_rms.mean) / torch.sqrt(agent.obs_rms.var + 1e-8), -10.0, 10.0)

                # 3. NORMALIZE EDİLMİŞ durumu kullanarak eylem üret
                action_dist = agent.actor(norm_state_tensor)
                action = action_dist.mean  # Testte deterministik eylem seç

            # Eylemi ortamda uygula
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            total_reward += reward

            # Simülasyonu yavaşlatmak için
            time.sleep(0.05)

        print(f"Bölüm {i + 1} tamamlandı. Toplam Ödül: {total_reward:.2f}")

    env.close()


if __name__ == '__main__':
    test()