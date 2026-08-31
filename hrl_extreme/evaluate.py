from .env import SparseGoalMazeEnv
from .agent import HierarchicalAgent

def evaluate_agent(episodes: int = 5):
    print("=" * 60)
    print("   HRL PROJECT EXTREME: TRAJECTORY EVALUATION & ROLLOUT   ")
    print("=" * 60)

    env = SparseGoalMazeEnv(grid_size=12, max_steps=100)
    agent = HierarchicalAgent(
        obs_dim=env.observation_dim,
        subgoal_dim=env.subgoal_dim,
        action_dim=env.action_space_n,
        c_step=8,
    )

    for ep in range(1, episodes + 1):
        obs = env.reset(seed=ep * 42)
        done = False
        step = 0
        positions = []

        print(f"\n--- Evaluation Rollout #{ep} ---")
        while not done:
            action, subgoal = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            positions.append([round(p, 2) for p in info["agent_pos"]])
            obs = next_obs
            step += 1

        print(f"  Result: {'[GOAL REACHED]' if info['is_success'] else '[MAX STEPS REACHED]'}")
        print(f"  Steps: {step} | Final Distance to Goal: {info['dist_to_goal']:.2f}")
        print(f"  Trajectory path points: {positions[:4]} ... {positions[-2:]}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    evaluate_agent()
