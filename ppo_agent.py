"""
PPO Agent Implementation
========================

Proximal Policy Optimization agent with:
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Observation and reward normalization
- Clipped surrogate objective
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and standard deviation.
    Memory-efficient as it doesn't store all historical data.
    """
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = 1e-4

    def update(self, x):
        """Update running statistics with new batch"""
        if x.shape[0] < 2:
            return  # Skip if batch too small

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
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
    """
    Policy network (actor) that outputs action distribution
    Uses tanh activation and diagonal Gaussian policy
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.max_action = max_action

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

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
        log_std = torch.clamp(self.log_std, -20, 2)
        log_std = log_std.expand_as(mean)

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class CriticNetwork(nn.Module):
    """
    Value network (critic) that estimates state value V(s)
    """
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
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
    """
    PPO Agent with Actor-Critic architecture

    Features:
    - Observation normalization using running statistics
    - Optional reward normalization
    - GAE for advantage estimation
    - Clipped surrogate objective
    - Early stopping based on KL divergence
    """
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr,
                 gamma, gae_lambda, clip_ratio, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.device = device

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        # Normalization statistics
        self.obs_rms = RunningMeanStd(shape=(state_dim,), device=device)
        self.reward_rms = RunningMeanStd(shape=(1,), device=device)
        self.normalize_rewards = True

    def to(self, device):
        """Move networks to device"""
        self.actor.to(device)
        self.critic.to(device)

    def save(self, directory, filename):
        """Save model and normalization statistics"""
        print("[SAVE] Saving models...")
        torch.save(self.actor.state_dict(), f'./{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'./{directory}/{filename}_critic.pth')
        torch.save({
            'mean': self.obs_rms.mean,
            'var': self.obs_rms.var,
            'count': self.obs_rms.count,
            'reward_mean': self.reward_rms.mean,
            'reward_var': self.reward_rms.var,
            'reward_count': self.reward_rms.count
        }, f'./{directory}/{filename}_rms.pth')

    def load(self, directory, filename):
        """Load model and normalization statistics"""
        print("[LOAD] Loading models...")
        self.actor.load_state_dict(torch.load(f'./{directory}/{filename}_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'./{directory}/{filename}_critic.pth', map_location=self.device))
        rms_data = torch.load(f'./{directory}/{filename}_rms.pth', map_location=self.device)
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
        self.obs_rms.count = rms_data['count']
        if 'reward_mean' in rms_data:
            self.reward_rms.mean = rms_data['reward_mean']
            self.reward_rms.var = rms_data['reward_var']
            self.reward_rms.count = rms_data['reward_count']

    def get_action(self, state):
        """
        Get action from current policy

        Args:
            state: Environment state

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Estimated state value
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        # Normalize observation
        norm_state = torch.clamp(
            (state - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8),
            -10.0, 10.0
        )

        with torch.no_grad():
            action_dist = self.actor(norm_state)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            value = self.critic(norm_state)

        return action, log_prob, value

    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        if not self.normalize_rewards:
            return reward

        if isinstance(reward, (int, float)):
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        else:
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.reward_rms.update(reward_tensor)
        normalized = reward / (torch.sqrt(self.reward_rms.var) + 1e-8)

        return normalized.item() if isinstance(reward, (int, float)) else normalized.flatten()

    def compute_advantages(self, rewards, dones, values, next_value):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)

        Args:
            rewards: List of rewards
            dones: List of done flags
            values: List of state values
            next_value: Value of next state

        Returns:
            advantages: Computed advantages
            returns: Computed returns (targets for value function)
        """
        device = next_value.device
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(device)
        values = torch.cat(values + [next_value], dim=0)

        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, 1).to(device)
        returns = torch.zeros(num_steps, 1).to(device)

        last_advantage = 0
        last_return = next_value

        # Backward computation of advantages
        for t in reversed(range(num_steps)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            last_return = values[t] + last_advantage

            advantages[t] = last_advantage
            returns[t] = last_return

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self, states, actions, log_probs, returns, advantages, num_epochs, batch_size, entropy_coef):
        """
        Update policy and value function using PPO

        Args:
            states: Batch of states
            actions: Batch of actions
            log_probs: Batch of log probabilities
            returns: Batch of returns
            advantages: Batch of advantages
            num_epochs: Number of update epochs
            batch_size: Mini-batch size
            entropy_coef: Entropy coefficient for exploration

        Returns:
            actor_loss: Mean actor loss
            critic_loss: Mean critic loss
            avg_std: Average action standard deviation
        """
        device = returns.device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)

        # Handle parallel environments
        if isinstance(actions, list):
            actions = torch.cat(actions, dim=0)
        if isinstance(log_probs, list):
            old_log_probs = torch.cat(log_probs, dim=0)
        else:
            old_log_probs = log_probs

        # Update observation normalization
        self.obs_rms.update(states)

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

                # Normalize observations
                norm_batch_states = torch.clamp(
                    (batch_states - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8),
                    -10.0, 10.0
                )

                # Actor loss
                dist = self.actor(norm_batch_states)
                std_devs.append(dist.stddev.mean().item())

                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    kl_divs.append(kl_div)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages.squeeze()

                entropy_loss = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy_loss

                # Critic loss
                new_values = self.critic(norm_batch_states).squeeze()
                critic_loss = 0.5 * nn.MSELoss()(new_values, batch_returns.squeeze())

                # Update networks
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

            # Early stopping if KL divergence too large
            mean_kl = np.mean(kl_divs[-len(range(0, num_samples, batch_size)):])
            if mean_kl > 0.02:
                print(f"[EARLY STOP] Epoch {epoch + 1}/{num_epochs} (KL={mean_kl:.4f})")
                break

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(std_devs)