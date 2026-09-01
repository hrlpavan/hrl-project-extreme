import time
import argparse
from typing import Optional
from .env import SparseGoalMazeEnv
from .continuous_env import ContinuousGoalNavigationEnv
from .vec_env import make_vec_env
from .agent import HierarchicalAgent

def train_hrl(
    mode: str = "continuous",
    num_episodes: int = 100,
    vec_envs: int = 4,
    c_step: int = 8,
    batch_size: int = 32,
    log_interval: int = 10,
    use_torch: bool = True,
):
    print("=" * 70)
    print("   HRL PROJECT EXTREME: HIGH-CAPACITY HIERARCHICAL REINFORCEMENT LEARNING   ")
    print("=" * 70)
    print(f"  Execution Mode       : {mode.upper()} CONTROL")
    print(f"  Parallel Rollout Envs: {vec_envs} workers")
    print(f"  Macro Step Interval  : C = {c_step}")
    print(f"  Target Episodes      : {num_episodes}")
    print(f"  PyTorch Optimization : {use_torch}")
    print("-" * 70)

    # Initialize environment factory
    is_continuous = (mode.lower() == "continuous")
    env_class = ContinuousGoalNavigationEnv if is_continuous else SparseGoalMazeEnv

    # Create parallel vectorized environment
    vec_env = make_vec_env(env_class, num_envs=vec_envs)

    agent = HierarchicalAgent(
        obs_dim=vec_env.observation_dim,
        subgoal_dim=vec_env.subgoal_dim,
        action_dim=vec_env.action_dim,
        continuous=is_continuous,
        c_step=c_step,
        use_torch=use_torch,
    )

    print(f"  Obs Dim: {vec_env.observation_dim} | Subgoal Dim: {vec_env.subgoal_dim} | Action Dim: {vec_env.action_dim}")
    print(f"  Agent Backend: {'PyTorch (' + str(agent.device) + ')' if agent.use_torch else 'Native Vectorized Math Engine'}\n")

    obs_list = vec_env.reset()
    episodes_completed = 0
    total_steps = 0
    recent_rewards = []
    recent_successes = []

    steps_in_subgoal = [0] * vec_envs
    current_subgoals = [[0.0] * vec_env.subgoal_dim for _ in range(vec_envs)]
    manager_start_obs = [list(obs) for obs in obs_list]
    cumulative_rewards = [0.0] * vec_envs

    t_start = time.time()
    t_last_log = t_start
    steps_last_log = 0

    while episodes_completed < num_episodes:
        # Batched action selection
        actions, current_subgoals, log_probs, values = agent.select_actions_vec(
            obs_list=obs_list,
            steps_in_subgoal=steps_in_subgoal,
            current_subgoals=current_subgoals,
            evaluate=False,
        )

        # Synchronous step across all vector workers
        next_obs_list, rewards, dones, infos = vec_env.step(actions)
        total_steps += vec_envs

        for i in range(vec_envs):
            steps_in_subgoal[i] += 1
            cumulative_rewards[i] += rewards[i]
            intrinsic_rew = agent.compute_intrinsic_reward(obs_list[i], next_obs_list[i], current_subgoals[i])

            # Store worker micro-transition
            agent.buffer.store_worker(
                obs=obs_list[i],
                subgoal=current_subgoals[i],
                action=actions[i],
                intrinsic_reward=intrinsic_rew,
                next_obs=next_obs_list[i],
                done=dones[i],
                log_prob=log_probs[i],
                value=values[i],
            )

            # Store manager macro-transition at boundary or termination
            if steps_in_subgoal[i] >= c_step or dones[i]:
                agent.buffer.store_manager(
                    obs=manager_start_obs[i],
                    subgoal=current_subgoals[i],
                    cumulative_extrinsic_reward=cumulative_rewards[i],
                    next_obs=next_obs_list[i],
                    done=dones[i],
                    log_prob=0.0,
                    value=0.0,
                )
                manager_start_obs[i] = list(next_obs_list[i])
                cumulative_rewards[i] = 0.0
                steps_in_subgoal[i] = 0

            if dones[i]:
                episodes_completed += 1
                recent_rewards.append(cumulative_rewards[i] + rewards[i])
                is_succ = infos[i].get("is_success", False)
                recent_successes.append(1 if is_succ else 0)

                # Reset manager tracking for this env
                manager_start_obs[i] = list(next_obs_list[i])
                cumulative_rewards[i] = 0.0
                steps_in_subgoal[i] = 0

                if episodes_completed % log_interval == 0:
                    t_now = time.time()
                    dt = t_now - t_last_log
                    fps = (total_steps - steps_last_log) / max(1e-5, dt)
                    avg_rew = sum(recent_rewards[-log_interval:]) / log_interval
                    sr = (sum(recent_successes[-log_interval:]) / log_interval) * 100.0

                    losses = agent.train_batch(batch_size=batch_size)
                    print(f"  Ep {episodes_completed:03d}/{num_episodes} | Avg Rew: {avg_rew:6.2f} | Success: {sr:5.1f}% | Steps: {total_steps:5d} | FPS: {fps:5.0f} | W-Loss: {losses['worker_loss']:.4f} | M-Loss: {losses['manager_loss']:.4f}")

                    t_last_log = t_now
                    steps_last_log = total_steps

        obs_list = next_obs_list

        # Periodic gradient update
        if total_steps % (batch_size * 2) == 0:
            agent.train_batch(batch_size=batch_size)

    total_time = time.time() - t_start
    final_sr = (sum(recent_successes[-20:]) / max(1, min(20, len(recent_successes)))) * 100.0
    overall_fps = total_steps / max(1e-5, total_time)

    print("\n" + "=" * 70)
    print(f"  [COMPLETED] Successfully trained {episodes_completed} episodes in {total_time:.2f}s")
    print(f"  [THROUGHPUT] Total Environment Steps: {total_steps} | Throughput: {overall_fps:.1f} FPS")
    print(f"  [EVALUATION] Final 20-Episode Success Rate: {final_sr:.1f}%")
    print("=" * 70 + "\n")

    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRL Project Extreme Training Engine")
    parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "discrete"], help="Control action mode")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--vec-envs", type=int, default=4, help="Number of parallel vectorized workers")
    parser.add_argument("--c-step", type=int, default=8, help="Macro manager step dilation interval")
    parser.add_argument("--no-torch", action="store_true", help="Force native math engine without PyTorch")
    args = parser.parse_args()

    train_hrl(
        mode=args.mode,
        num_episodes=args.episodes,
        vec_envs=args.vec_envs,
        c_step=args.c_step,
        use_torch=not args.no_torch,
    )
