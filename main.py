import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import numpy as np
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan Cihaz: {device}")

def main():
    # --- Hiperparametreler ---
    total_timesteps = 6000000   # Toplam eğitim adımı sayısı
    # learning_rate = 3e-4      # Öğrenme oranı
    actor_learning_rate = 3e-4  # Aktör öğrenme oranı
    critic_learning_rate = 3e-4 # Kritik için daha düşük bir öğrenme oranı
    gamma = 0.99                # Gelecekteki ödülleri ne kadar önemseyeceğimizi belirleyen indirim faktörü
    gae_lambda = 0.95           # GAE için lambda değeri (Avantaj hesaplamada kullanılır)
    rollout_steps = 2048        # Her bir güncelleme öncesi toplanacak veri (adım) sayısı
    update_epochs = 1           # Toplanan veri ile sinir ağlarının kaç defa güncelleneceği
    clip_ratio = 0.2            # PPO'nun "clipped" kayıp fonksiyonu için kırpma oranı
    batch_size = 64             # Her bir güncelleme adımında kullanılacak veri yığını boyutu
    entropy_coef = 0.01         # Entropy coefficient

    # --- Deney Ayarları ---
    load_model = False
    experiment_name = "Ant-v5_PPO_T_***"
    base_experiment_name = "Ant-v5_PPO_T"
    run_name = f"{base_experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./runs"):
        os.makedirs("./runs")

    log_file_path = f"./runs/{run_name}.txt"
    writer = SummaryWriter(f"./runs/{run_name}")

    def log_to_file(message):
        with open(log_file_path, 'a') as f:
            f.write(message + '\\n')

    # 1. Ortamı Yükle
    env = gym.make('Ant-v5')

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
    agent = PPOAgent(state_dim, action_dim, max_action, actor_learning_rate, critic_learning_rate, gamma, gae_lambda, clip_ratio, device)
    agent.to(device)

    def lr_lambda(step):
        # total_timesteps'e ulaşıldığında öğrenme oranı 0 olacak şekilde lineer azalt
        return max(0.0, 1.0 - (step / total_timesteps))

    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.actor_optimizer, lr_lambda)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.critic_optimizer, lr_lambda)

    if load_model:
        try:
            agent.load("models", experiment_name)
            log_to_file("Önceden eğitilmiş model yüklendi.")
        except FileNotFoundError:
            log_to_file("Model dosyası bulunamadı, sıfırdan eğitime başlanıyor.")

    # Ortamı sıfırla ve ilk durumu (state) al
    state, info = env.reset()

    # Toplam adım sayısını takip etmek için sayaç
    global_step_count = 0

    # Loglama için
    current_episode_reward = 0
    episode_rewards_list = []

    # Ana Eğitim Döngüsü
    while global_step_count < total_timesteps:

        # --- 1. Veri Toplama (Rollout) Aşaması ---
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(rollout_steps):
            global_step_count += 1
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

            with torch.no_grad():
                action_dist = agent.actor(state_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=1)
                value = agent.critic(state_tensor)

            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state
            current_episode_reward += reward

            if done:
                episode_rewards_list.append(current_episode_reward)
                avg_reward = np.mean(episode_rewards_list[-100:])
                print(
                    f"Adım: {global_step_count}, Bölüm Bitti, Ödül: {current_episode_reward:.2f}, Son 100 Bölüm Ort. Ödül: {avg_reward:.2f}")
                current_episode_reward = 0
                state, info = env.reset()

                log_message = f"Adım: {global_step_count}, Ödül: {current_episode_reward:.2f}, Ort. Ödül: {avg_reward:.2f}"
                print(log_message)
                log_to_file(log_message)

        # --- 2. Avantaj ve Getiri Hesaplama ---
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            next_value = agent.critic(next_state_tensor)
        advantages, returns = agent.compute_advantages(rewards, dones, values, next_value)

        # --- 3. Öğrenme Aşaması ---
        actor_loss, critic_loss, avg_std = agent.learn(states, actions, log_probs, returns, advantages, update_epochs, batch_size, entropy_coef)

        actor_scheduler.step()
        critic_scheduler.step()

        # --- 4. TensorBoard'a Loglama ---
        writer.add_scalar('losses/actor_loss', actor_loss, global_step_count)
        writer.add_scalar('losses/critic_loss', critic_loss, global_step_count)
        writer.add_scalar('charts/avg_reward_100_episodes', np.mean(episode_rewards_list[-100:]), global_step_count)
        writer.add_scalar('charts/exploration_std', avg_std, global_step_count)
        writer.add_scalar('charts/learning_rate', actor_scheduler.get_last_lr()[0], global_step_count)

    agent.save("models", run_name)
    log_to_file("Eğitim tamamlandı ve modeller kaydedildi.")
    writer.close()
    env.close()
    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    main()