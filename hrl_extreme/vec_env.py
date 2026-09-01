import copy
from typing import List, Tuple, Dict, Any, Callable, Optional, Union

class VectorEnv:
    """
    Parallel Vectorized Environment Manager for Hierarchical Reinforcement Learning.
    Synchronously steps N independent environment instances, auto-resets terminated environments,
    and returns batched states, rewards, dones, and diagnostic info.
    """
    def __init__(self, env_fns: List[Callable[[], Any]]):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]

        # Cache environment properties
        first_env = self.envs[0]
        self.observation_dim = getattr(first_env, "observation_dim", 4)
        self.action_dim = getattr(first_env, "action_dim", getattr(first_env, "action_space_n", 4))
        self.subgoal_dim = getattr(first_env, "subgoal_dim", 2)
        self.continuous = getattr(first_env, "continuous", False)

        # Per-environment macro counters & states
        self.steps_in_subgoal = [0] * self.num_envs
        self.current_subgoals = [[0.0] * self.subgoal_dim for _ in range(self.num_envs)]
        self.manager_start_obs = [None] * self.num_envs
        self.cumulative_rewards = [0.0] * self.num_envs

    def reset(self, seeds: Optional[List[int]] = None) -> List[List[float]]:
        obs_list = []
        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            obs = env.reset(seed=seed)
            obs_list.append(obs)
            self.steps_in_subgoal[i] = 0
            self.manager_start_obs[i] = list(obs)
            self.cumulative_rewards[i] = 0.0
        return obs_list

    def step(self, actions: List[Union[int, List[float]]]) -> Tuple[
        List[List[float]], List[float], List[bool], List[Dict[str, Any]]
    ]:
        """
        Step all environments in lockstep.
        Auto-resets any environment that completes.
        """
        next_obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            dones.append(done)

            info_copy = dict(info)
            info_copy["terminal_observation"] = list(next_obs) if done else None
            infos.append(info_copy)

            if done:
                # Auto-reset environment
                reset_obs = env.reset()
                next_obs_list.append(reset_obs)
            else:
                next_obs_list.append(next_obs)

        return next_obs_list, rewards, dones, infos

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

def make_vec_env(env_class: Callable, num_envs: int = 4, **kwargs) -> VectorEnv:
    """
    Factory helper to create a VectorEnv of `num_envs` instances.
    """
    env_fns = [lambda: env_class(**kwargs) for _ in range(num_envs)]
    return VectorEnv(env_fns)
