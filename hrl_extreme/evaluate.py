import argparse
from typing import Optional
from .env import SparseGoalMazeEnv
from .continuous_env import ContinuousGoalNavigationEnv
from .agent import HierarchicalAgent

def evaluate_agent(
    mode: str = "continuous",
    episodes: int = 5,
    c_step: int = 8,
    use_torch: bool = True,
):
    print("=" * 70)
    print("   HRL PROJECT EXTREME: TRAJECTORY EVALUATION & ROLLOUT BENCHMARK   ")
    print("=" * 70)
    print(f"  Mode           : {mode.upper()} CONTROL")
    print(f"  Target Episodes: {episodes}")
    print(f"  Macro Interval : C = {c_step}")
    print("-" * 70)

    is_continuous = (mode.lower() == "continuous")
    env = ContinuousGoalNavigationEnv() if is_continuous else SparseGoalMazeEnv()

    agent = HierarchicalAgent(
        obs_dim=env.observation_dim,
        subgoal_dim=env.subgoal_dim,
        action_dim=env.action_dim if is_continuous else env.action_space_n,
        continuous=is_continuous,
        c_step=c_step,
        use_torch=use_torch,
    )

    success_count = 0

    for ep in range(1, episodes + 1):
        obs = env.reset(seed=ep * 42)
        done = False
        step = 0
        trajectory = []
        subgoal_history = []
        total_extrinsic_reward = 0.0

        print(f"\n--- Evaluation Rollout #{ep} ---")
        while not done:
            action, subgoal, _, _ = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)

            total_extrinsic_reward += reward
            pos = [round(p, 3) for p in info.get("agent_pos", obs[:2])]
            trajectory.append(pos)
            if step % c_step == 0:
                subgoal_history.append([round(g, 3) for g in subgoal])

            obs = next_obs
            step += 1

        is_succ = info.get("is_success", False)
        if is_succ:
            success_count += 1

        result_tag = "[GOAL REACHED]" if is_succ else "[MAX STEPS REACHED]"
        print(f"  Result          : {result_tag}")
        print(f"  Total Steps     : {step} / {env.max_steps}")
        print(f"  Cumulative Rew  : {total_extrinsic_reward:.2f}")
        print(f"  Final Distance  : {info.get('dist_to_goal', 0.0):.3f}")
        print(f"  Macro Sub-Goals : {subgoal_history[:3]} ... {subgoal_history[-1:]}")
        print(f"  Trajectory Path : {trajectory[:3]} ... {trajectory[-2:]}")

    sr = (success_count / episodes) * 100.0
    print("\n" + "=" * 70)
    print(f"  [BENCHMARK] Success Rate: {sr:.1f}% ({success_count}/{episodes} runs)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRL Project Extreme Trajectory Evaluation")
    parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--c-step", type=int, default=8)
    parser.add_argument("--no-torch", action="store_true")
    args = parser.parse_args()

    evaluate_agent(
        mode=args.mode,
        episodes=args.episodes,
        c_step=args.c_step,
        use_torch=not args.no_torch,
    )
