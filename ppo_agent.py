import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        value = self.value_layer(x)
        return value


# PPO Ajan Sınıfı ağları, optimizer'ları ve öğrenme mantığını bir araya getiriyor.
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, lr, gamma, gae_lambda, clip_ratio):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio

        # Aktör ve Kritik ağlarını oluştur
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim)

        # Optimizer'ları oluştur
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def compute_advantages(self, rewards, dones, values, next_value):
        """
        GAE (Generalized Advantage Estimation) kullanarak avantajları hesaplar.
        Aynı zamanda, Kritik ağını eğitmek için kullanılacak olan getirileri (returns) de hesaplar.
        """

        # 'rewards', 'dones', 'values' listelerini PyTorch tensörlerine dönüştür
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
        values = torch.cat(values + [next_value], dim=0)

        # Avantajları ve getirileri saklamak için tensörler oluştur
        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, 1)
        returns = torch.zeros(num_steps, 1)

        # GAE hesaplaması sondan başa doğru yapılır
        last_advantage = 0
        last_return = next_value  # Döngüye başlarken son adımdaki getiri, bir sonraki durumun değeridir

        for t in reversed(range(num_steps)):
            # 'done' ise, bir sonraki durumun değeri 0'dır, çünkü bölüm bitmiştir.
            mask = 1.0 - dones[t]

            # TD Hatası (TD Error): Gerçekleşen ödül + bir sonraki durumun değeri - mevcut durumun değeri
            # Bu, tek bir adımda beklentimizin ne kadar saptığını gösterir.
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]

            # GAE: Mevcut TD hatası + bir sonraki adımdan gelen (indirgenmiş) avantaj
            # Bu, TD hatasını birden çok adıma geneller.
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask

            # Getiri (Return): Mevcut durumun değerine hesaplanan avantajı ekleriz.
            # Kritik ağının öğrenmesi gereken hedef değer budur.
            last_return = values[t] + last_advantage

            advantages[t] = last_advantage
            returns[t] = last_return

        return advantages, returns

    def learn(self, states, actions, log_probs, returns, advantages, num_epochs, batch_size):
        """
        Toplanan verileri kullanarak Aktör ve Kritik ağlarını günceller.
        """

        # Listeleri tek bir büyük tensöre dönüştür. Bu, verimli batch işleme için gereklidir.
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(log_probs, dim=0)

        # Her bir güncelleme döngüsü (epoch) için...
        for _ in range(num_epochs):

            # Veri setini karıştırıp küçük yığınlara (batches) bölerek işleyeceğiz.
            # Bu, öğrenme sürecini daha stabil hale getirir.
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Mevcut yığın için verileri seç
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # --- 1. AKTÖR KAYBINI (LOSS) HESAPLA ---

                # Mevcut (güncellenmiş) politika ile eylemlerin log olasılıklarını yeniden hesapla
                dist = self.actor(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # Oran (Ratio): r_t(θ) = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
                # Yeni politika ile eski politika ne kadar farklı?
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Kırpılmış (Clipped) Amaç Fonksiyonu
                # PPO'nun kalbi burasıdır.
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages

                # İki versiyondan daha "kötümser" olanını seçerek büyük güncellemeleri engelleriz.
                # Gradient ascent yapmak istediğimiz için kaybı negatif yapıyoruz.
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- 2. KRİTİK KAYBINI (LOSS) HESAPLA ---

                # Kritik ağının mevcut tahminleri ile hedeflenen getiriler (returns) arasındaki fark.
                # Mean Squared Error (Ortalama Kare Hata)
                new_values = self.critic(batch_states)
                critic_loss = nn.MSELoss()(new_values, batch_returns)

                # --- 3. AĞLARI GÜNCELLE ---

                # Aktör Güncellemesi
                self.actor_optimizer.zero_grad()  # Önceki gradient'leri sıfırla
                actor_loss.backward()  # Yeni gradient'leri hesapla
                self.actor_optimizer.step()  # Ağırlıkları güncelle

                # Kritik Güncellemesi
                self.critic_optimizer.zero_grad()
                # 0.5 ile çarpmak yaygın bir pratiktir, iki loss'un büyüklüğünü dengeler.
                (0.5 * critic_loss).backward()
                self.critic_optimizer.step()