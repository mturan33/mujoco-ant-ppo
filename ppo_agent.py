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
        # 🔥 BÜYÜTÜLMÜŞ MİMARİ - Ant için optimize
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.max_action = max_action

        # Öğrenilebilir log_std (action başına)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Orthogonal initialization
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))

        # Log std'yi sınırla (-20, 2) -> std aralığı: (2e-9, 7.4)
        log_std = torch.clamp(self.log_std, -20, 2)
        log_std = log_std.expand_as(mean)

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        # 🔥 BÜYÜTÜLMÜŞ MİMARİ - Actor ile dengeli
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_layer = nn.Linear(128, 1)

        # Orthogonal initialization
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.value_layer.weight, gain=1.0)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.value_layer(x)
        return value


class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr, gamma, gae_lambda, clip_ratio, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.device = device

        # Ağları oluştur
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim)

        # Optimizer'lar - Critic için weight decay ekledik
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        # Observation normalization
        self.obs_rms = RunningMeanStd(shape=(state_dim,), device=device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def save(self, directory, filename):
        print("[SAVE] Saving models...")
        torch.save(self.actor.state_dict(), f'./{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'./{directory}/{filename}_critic.pth')
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count
        }, f'./{directory}/{filename}_rms.pth')

    def load(self, directory, filename):
        print("[LOAD] Loading models...")
        self.actor.load_state_dict(torch.load(f'./{directory}/{filename}_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'./{directory}/{filename}_critic.pth', map_location=self.device))
        rms_data = torch.load(f'./{directory}/{filename}_rms.pth', map_location=self.device)
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
        self.obs_rms.count = rms_data['count']

    def compute_advantages(self, rewards, dones, values, next_value):
        """GAE ile avantaj hesaplama"""
        device = next_value.device
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(device)
        values = torch.cat(values + [next_value], dim=0)

        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, 1).to(device)
        returns = torch.zeros(num_steps, 1).to(device)

        last_advantage = 0
        last_return = next_value

        for t in reversed(range(num_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            last_return = values[t] + last_advantage

            advantages[t] = last_advantage
            returns[t] = last_return

        return advantages, returns

    def learn(self, states, actions, log_probs, returns, advantages, num_epochs, batch_size, entropy_coef):
        device = returns.device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(log_probs, dim=0)

        # Observation normalization'ı güncelle
        self.obs_rms.update(states)

        # 🔥 Avantaj normalizasyonu
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []
        std_devs = []
        kl_divs = []

        # Mini-batch SGD
        for epoch in range(num_epochs):
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

                # 🔥 Observation normalization
                norm_batch_states = torch.clamp(
                    (batch_states - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8),
                    -10.0, 10.0
                )

                # --- ACTOR LOSS ---
                dist = self.actor(norm_batch_states)
                std_devs.append(dist.stddev.mean().item())

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # KL divergence için (early stopping check)
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    kl_divs.append(kl_div)

                surr1 = ratio * batch_advantages.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages.squeeze()

                # Entropy bonus (keşfi teşvik eder)
                entropy_loss = dist.entropy().mean()

                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy_loss

                # --- CRITIC LOSS (Basit MSE) ---
                new_values = self.critic(norm_batch_states).squeeze()
                critic_loss = nn.MSELoss()(new_values, batch_returns.squeeze())

                # --- BACKPROP ---
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

            # Early stopping eğer KL divergence çok büyükse
            mean_kl = np.mean(kl_divs[-len(range(0, num_samples, batch_size)):])
            if mean_kl > 0.02:  # Target KL
                print(f"[EARLY STOP] Epoch {epoch + 1}/{num_epochs} (KL={mean_kl:.4f})")
                break

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(std_devs)