import gymnasium as gym

env = gym.make('Ant-v5')
print(f"State dim: {env.observation_space.shape[0]}")

# Sample episode
obs, info = env.reset()
print(f"\nİlk 10 state değeri:")
print(obs[:10])

# 5 step at
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Height (obs[0]): {obs[0]:.3f}, obs[2]: {obs[2]:.3f}")