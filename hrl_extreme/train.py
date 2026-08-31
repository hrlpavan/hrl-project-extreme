import time
from .env import SparseGoalMazeEnv
from .agent import HierarchicalAgent

def train_hrl(num_episodes: int = 100, log_interval: int = 10):
    print("=" * 65)
    print(" 🧠 HRL PROJECT EXTREME: HIERARCHICAL REINFORCEMENT LEARNING ")
    print("=" * 65)

    env = SparseGoalMazeEnv(grid_size=12, max_steps=100)
    agent = HierarchicalAgent(
        obs_dim=env.observation_dim,
        subgoal_dim=env.subgoal_dim,
        action_dim=env.action_space_n,
        c_step=8,
    )

    print(f"Environment: Sparse-Reward Grid ({env.grid_size}x{env.grid_size})")
    print(f"Observation Dim: {env.observation_dim} | Sub-Goal Dim: {env.subgoal_dim} | Actions: {env.action_space_n}")
    print(f"Manager Macro-Step Interval: C = {agent.c_step}\n")

    success_history = []
    reward_history = []
    t_start = time.time()

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, subgoal = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.step_update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            steps += 1

        losses = agent.train_batch(batch_size=16)

        is_success = info.get("is_success", False)
        success_history.append(1 if is_success else 0)
        reward_history.append(ep_reward)

        if ep % log_interval == 0:
            avg_rew = sum(reward_history[-log_interval:]) / log_interval
            sr = (sum(success_history[-log_interval:]) / log_interval) * 100.0
            print(f"Episode {ep:03d}/{num_episodes} | Avg Reward: {avg_rew:6.2f} | Success Rate: {sr:5.1f}% | Steps: {steps:3d} | W-Loss: {losses['worker_loss']:.4f} | M-Loss: {losses['manager_loss']:.4f}")

    total_time = time.time() - t_start
    final_sr = (sum(success_history[-20:]) / min(20, len(success_history))) * 100.0
    print("\n" + "=" * 65)
    print(f"  ✓ Training completed in {total_time:.2f}s across {num_episodes} episodes")
    print(f"  ✓ Final 20-Episode Success Rate: {final_sr:.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    train_hrl()
