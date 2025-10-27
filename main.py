import gymnasium as gym
import torch
from ppo_agent import ActorNetwork, CriticNetwork


def main():
    # 1. Ortamı Yükle
    env = gym.make('Ant-v4')

    # 2. Ortamdan Gerekli Bilgileri Al
    # Gözlem uzayının boyutu (state_dim)
    state_dim = env.observation_space.shape[0]

    # Eylem uzayının boyutu (action_dim)
    action_dim = env.action_space.shape[0]

    # Eylemlerin alabileceği maksimum değer (genellikle 1.0)
    # Tork değerleri genellikle -max_action ile +max_action arasındadır.
    max_action = float(env.action_space.high[0])

    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Max Action Value: {max_action}")

    # 3. Aktör ve Kritik Ağlarını Oluştur
    actor = ActorNetwork(state_dim, action_dim, max_action)
    critic = CriticNetwork(state_dim)

    # 4. Temel Etkileşim Döngüsü (Henüz Eğitim Yok)
    # Sadece ağların çalışıp çalışmadığını ve ortamla nasıl etkileşime girdiğini görelim.
    state, info = env.reset()

    # Durumu (state) bir PyTorch tensörüne dönüştür
    state_tensor = torch.FloatTensor(state.reshape(1, -1))

    # Aktör ağından bir olasılık dağılımı al
    action_dist = actor(state_tensor)

    # Dağılımdan bir eylem örnekle
    action = action_dist.sample()

    # Eylemin log-olasılığını hesapla (PPO'nun loss fonksiyonu için gerekecek)
    log_prob = action_dist.log_prob(action)

    # Kritik ağından durumun değerini tahmin et
    state_value = critic(state_tensor)

    print("\\n--- Test Aşaması ---")
    print(f"Örnek Eylem (Action): {action.cpu().detach().numpy()}")
    print(f"Eylemin Log Olasılığı: {log_prob.cpu().detach().numpy()}")
    print(f"Durumun Değer Tahmini (V(s)): {state_value.cpu().detach().numpy()}")

    env.close()


if __name__ == '__main__':
    main()