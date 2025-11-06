"""
2x2 Grid Demo - Compare 4 Models Side-by-Side
==============================================

Displays 4 trained PPO models simultaneously in a 2x2 grid layout.
Each quadrant shows model visualization, current score, and max score.

Controls:
- 'q' or ESC: Quit
- 's': Save screenshot
"""

import gymnasium as gym
import torch
from ppo_agent import PPOAgent
import numpy as np
import cv2
import time
from datetime import datetime


class GridDemo:
    def __init__(self, model_configs):
        """
        Initialize 2x2 grid demo with 4 models

        Args:
            model_configs: List of dicts with 'name' and 'path' keys
            Example: [{'name': 'Model A', 'path': 'model_path'}, ...]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = len(model_configs)
        assert self.num_models == 4, "Exactly 4 models required for 2x2 grid"

        self.envs = []
        self.agents = []
        self.model_names = []

        print(f"[INFO] Loading {self.num_models} models...")

        for i, config in enumerate(model_configs):
            # Create environment with shorter episodes
            env = gym.make('Ant-v5', render_mode='rgb_array', max_episode_steps=200)
            self.envs.append(env)

            # Create agent
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

            agent = PPOAgent(
                state_dim, action_dim, max_action,
                actor_lr=3e-4, critic_lr=3e-4,
                gamma=0.99, gae_lambda=0.95,
                clip_ratio=0.2, device=self.device
            )
            agent.to(self.device)

            # Load trained model
            try:
                agent.load("models", config['path'])
                print(f"[OK] Loaded: {config['name']}")
            except FileNotFoundError:
                print(f"[ERROR] Model not found: {config['path']}")
                print(f"[INFO] Using random policy for {config['name']}")

            self.agents.append(agent)
            self.model_names.append(config['name'])

        # Grid display settings
        self.cell_width = 480
        self.cell_height = 360
        self.grid_width = self.cell_width * 2
        self.grid_height = self.cell_height * 2 + 100

        # Video recording
        self.video_writer = None

    def render_cell(self, env, agent, model_name, current_score, max_score, cell_img):
        """Render single cell with model visualization and stats"""
        frame = env.render()
        frame_resized = cv2.resize(frame, (self.cell_width, self.cell_height - 80))

        # Create text area
        text_area = np.ones((80, self.cell_width, 3), dtype=np.uint8) * 40
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Add model info
        cv2.putText(text_area, model_name, (10, 25),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(text_area, f"Score: {current_score:.1f}", (10, 50),
                    font, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(text_area, f"Max: {max_score:.1f}", (10, 70),
                    font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

        cell_img[:] = np.vstack([frame_resized, text_area])
        return cell_img

    def run(self, num_episodes=5, save_video=True):
        """Run demo for specified number of episodes"""

        # Initialize
        states = [env.reset()[0] for env in self.envs]
        current_scores = [0.0] * self.num_models
        max_scores = [0.0] * self.num_models
        episode_counts = [0] * self.num_models

        # Create window
        cv2.namedWindow('PPO Demo - 4 Models', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('PPO Demo - 4 Models', self.grid_width, self.grid_height)

        # Video recording setup
        if save_video:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = f'demo_2x2_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, 30.0,
                (self.grid_width, self.grid_height)
            )
            print(f"[RECORD] Saving to: {video_path}")

        print("\n" + "=" * 60)
        print("[START] 2x2 Grid Demo")
        print("=" * 60)
        print("Press 'q' to quit | Press 's' to save screenshot")
        print("=" * 60 + "\n")

        step_count = 0
        start_time = time.time()

        while True:
            if all(count >= num_episodes for count in episode_counts):
                print("\n[COMPLETE] All models finished!")
                break

            # Create grid canvas
            grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

            # Title bar
            title_bar = np.ones((100, self.grid_width, 3), dtype=np.uint8) * 30
            cv2.putText(title_bar, "PPO Ant-v5 Locomotion - Model Comparison",
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(title_bar, f"Step: {step_count} | FPS: {step_count / (time.time() - start_time):.1f}",
                        (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Process each model
            for i in range(self.num_models):
                if episode_counts[i] >= num_episodes:
                    continue

                # Get action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(states[i].reshape(1, -1)).to(self.device)
                    norm_state = torch.clamp(
                        (state_tensor - self.agents[i].obs_rms.mean) /
                        torch.sqrt(self.agents[i].obs_rms.var + 1e-8),
                        -10.0, 10.0
                    )
                    action_dist = self.agents[i].actor(norm_state)
                    action = action_dist.mean

                # Step environment
                next_state, reward, terminated, truncated, info = self.envs[i].step(
                    action.cpu().numpy().flatten()
                )
                done = terminated or truncated

                current_scores[i] += reward
                states[i] = next_state

                # Handle episode end
                if done:
                    episode_counts[i] += 1
                    max_scores[i] = max(max_scores[i], current_scores[i])

                    print(f"[{self.model_names[i]}] Episode {episode_counts[i]}/{num_episodes} | "
                          f"Score: {current_scores[i]:.1f} | Max: {max_scores[i]:.1f}")

                    current_scores[i] = 0.0
                    states[i] = self.envs[i].reset()[0]

            # Render cells
            cells = []
            for i in range(self.num_models):
                cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)
                self.render_cell(
                    self.envs[i],
                    self.agents[i],
                    self.model_names[i],
                    current_scores[i],
                    max_scores[i],
                    cell
                )
                cells.append(cell)

            # Arrange 2x2 grid
            top_row = np.hstack([cells[0], cells[1]])
            bottom_row = np.hstack([cells[2], cells[3]])
            grid_content = np.vstack([top_row, bottom_row])

            grid[0:100] = title_bar
            grid[100:] = grid_content

            # Display
            cv2.imshow('PPO Demo - 4 Models', grid)

            if self.video_writer:
                self.video_writer.write(grid)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[QUIT] User requested quit")
                break
            elif key == ord('s'):
                screenshot_path = f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                cv2.imwrite(screenshot_path, grid)
                print(f"[SAVED] Screenshot: {screenshot_path}")

            step_count += 1
            time.sleep(0.01)

        # Cleanup
        print("\n[CLEANUP] Closing environments...")
        for env in self.envs:
            env.close()

        if self.video_writer:
            self.video_writer.release()
            print("[SAVED] Video saved!")

        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        for i in range(self.num_models):
            print(f"{self.model_names[i]}")
            print(f"  Episodes: {episode_counts[i]}")
            print(f"  Max Score: {max_scores[i]:.1f}")
            print()


def main():
    """Main demo function with model configuration"""

    # Configure 4 models for comparison
    # Replace paths with your trained model names
    model_configs = [
        {
            'name': 'Vanilla BEST (2739)',
            'path': 'Ant-v5_PPO_OPTIMIZED_12envs_2025-11-05_12-05-21_BEST'
        },
        {
            'name': 'Anti-Hopping (2531)',
            'path': 'Ant-v5_PPO_ANTIHOPPING_16envs_2025-11-05_18-09-09_BEST'
        },
        {
            'name': 'Learning Phase (1828)',
            'path': 'Ant-v5_PPO_MINIMAL_2025-11-03_13-20-59_BEST'
        },
        {
            'name': 'Energy Efficient (1261)',
            'path': 'Ant-v5_PPO_ENERGY_EFFICIENT_2025-11-04_14-23-01_BEST'
        }
    ]

    print("=" * 60)
    print("2x2 GRID DEMO CONFIGURATION")
    print("=" * 60)
    for i, config in enumerate(model_configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   Path: models/{config['path']}")
    print("=" * 60 + "\n")

    # Create and run demo
    demo = GridDemo(model_configs)
    demo.run(num_episodes=5, save_video=True)

    print("\n[DONE] Demo complete!")


if __name__ == '__main__':
    main()