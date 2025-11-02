import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = 1e-4

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.max_action = max_action

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        torch.nn.init.orthogonal_(self.layer_1.weight)
        torch.nn.init.orthogonal_(self.layer_2.weight)
        torch.nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)


    def forward(self, state):
        x = torch.tanh(self.layer_1(state))
        x = torch.tanh(self.layer_2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))

        log_std = torch.clamp(self.log_std, -5, 2) # min=-5, max=2

        # log_std'yi batch boyutuna uyacak şekilde genişletiyoruz.
        log_std = log_std.expand_as(mean)

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 1)
        torch.nn.init.orthogonal_(self.layer_1.weight)
        torch.nn.init.orthogonal_(self.layer_2.weight)
        torch.nn.init.orthogonal_(self.value_layer.weight)

    def forward(self, state):
        x = torch.tanh(self.layer_1(state))
        x = torch.tanh(self.layer_2(x))
        value = self.value_layer(x)
        return value


# PPO Ajan Sınıfı ağları, optimizer'ları ve öğrenme mantığını bir araya getiriyor.
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, gamma, gae_lambda, clip_ratio, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio

        # Aktör ve Kritik ağlarını oluştur
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim)

        # Optimizer'ları oluştur
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.001)

        self.obs_rms = RunningMeanStd(shape=(state_dim,), device=device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def save(self, directory, filename):
        print("... Modeller kaydediliyor ...")
        torch.save(self.actor.state_dict(), f'./{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'./{directory}/{filename}_critic.pth')
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count
        }, f'./{directory}/{filename}_rms.pth')

    def load(self, directory, filename):
        print("... Modeller yükleniyor ...")
        self.actor.load_state_dict(torch.load(f'./{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'./{directory}/{filename}_critic.pth'))
        rms_data = torch.load(f'./{directory}/{filename}_rms.pth')
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
        self.obs_rms.count = rms_data['count']

    def compute_advantages(self, rewards, dones, values, next_value):
        """
        GAE (Generalized Advantage Estimation) kullanarak avantajları hesaplar.
        Aynı zamanda, Kritik ağını eğitmek için kullanılacak olan getirileri (returns) de hesaplar.
        """

        # 'rewards', 'dones', 'values' listelerini PyTorch tensörlerine dönüştür
        device = next_value.device # Gelen tensörden cihazı öğren
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(device)
        values = torch.cat(values + [next_value], dim=0)

        # Avantajları ve getirileri saklamak için tensörler oluştur
        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, 1).to(device)
        returns = torch.zeros(num_steps, 1).to(device)

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

    def learn(self, states, actions, log_probs, returns, advantages, num_epochs, batch_size, entropy_coef):
        # Veriyi ve istatistikleri hazırla
        device = returns.device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(log_probs, dim=0)

        self.obs_rms.update(states)

        actor_losses = []
        critic_losses = []
        std_devs = []

        # Her bir güncelleme döngüsü (epoch) için
        for _ in range(num_epochs):
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # --- 1. Avantaj Normalizasyonu ---
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # --- 2. Gözlem Normalizasyonu ---
                norm_batch_states = torch.clamp(
                    (batch_states - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)

                # --- 3. AKTÖR KAYBINI HESAPLA ---
                dist = self.actor(norm_batch_states)
                std_devs.append(dist.stddev.mean().item())
                entropy_loss = dist.entropy().mean()

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages

                actor_loss = -torch.min(surr1,
                                        surr2).mean() - entropy_coef * entropy_loss  # NOT: Entropiyi eksiye çektik, keşfi teşvik eder!

                # --- 4. KRİTİK KAYBINI HESAPLA (DEĞER KIRPMA İLE) ---
                new_values = self.critic(norm_batch_states).view(-1)

                # --- YENİ: Değer Fonksiyonu Kırpma (Value Clipping) ---
                # Kritik ağının tahminlerinin her güncellemede çok fazla değişmesini engeller.
                # Bu, Aktör'ün istikrarına da yardımcı olur.

                # 1. Önceki değerleri alıyoruz. Getiriler (returns) ve avantajlar (advantages)
                # hesaplandığı andaki değerler, returns - advantages formülüyle elde edilebilir.
                # Bu, kritik için "eski" bir tahmin oluşturur.
                old_values = (batch_returns - batch_advantages).view(-1)

                # 2. Değerleri, eski değerlerin etrafında küçük bir aralığa kırpıyoruz.
                values_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_ratio, self.clip_ratio
                )

                # 3. Kırpılmış ve kırpılmamış kayıpları hesaplıyoruz.
                critic_loss_unclipped = nn.MSELoss()(new_values, batch_returns.view(-1))
                critic_loss_clipped = nn.MSELoss()(values_clipped, batch_returns.view(-1))

                # 4. İki kayıptan daha büyük olanı seçerek "en kötü durum" senaryosuna göre
                # güncelleme yapıyoruz. Bu, güncellemeleri daha muhafazakar hale getirir.
                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

                # --- 5. AĞLARI GÜNCELLE (GRADYAN KIRPMA İLE) ---
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # --- YENİ: Gradyan Kırpma (Gradient Clipping) ---
                # Gradiyanların "patlamasını" ve politikayı bozmasını engeller.
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (0.5 * critic_loss).backward()
                # --- YENİ: Gradyan Kırpma (Gradient Clipping) ---
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # Ortalama kayıpları ve std'yi geri döndür ---
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(std_devs)