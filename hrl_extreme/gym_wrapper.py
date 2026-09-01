import math
from typing import Dict, Tuple, Any, List, Optional, Union

class GymEnvWrapper:
    """
    Standard adapter for Gymnasium and MuJoCo environments.
    Converts Gym observation/action spaces to flat continuous vectors suitable for
    Hierarchical Reinforcement Learning policies.
    """
    def __init__(self, env_id: str = "PointMaze_UMaze-v3", max_steps: Optional[int] = None):
        self.env_id = env_id
        self.gym_env = None
        self._is_gym_available = False

        try:
            import gymnasium as gym
            self.gym_env = gym.make(env_id)
            self._is_gym_available = True
        except Exception:
            # Fallback or stub if gymnasium / mujoco is not installed
            self.gym_env = None

        if self.gym_env is not None:
            # Extract observation dim
            obs_space = self.gym_env.observation_space
            if hasattr(obs_space, "shape") and obs_space.shape is not None:
                self.observation_dim = int(obs_space.shape[0])
            elif isinstance(obs_space, dict) or hasattr(obs_space, "spaces"):
                # Dict space (e.g., GoalEnv: observation, achieved_goal, desired_goal)
                self.observation_dim = sum(int(s.shape[0]) for s in obs_space.spaces.values())
            else:
                self.observation_dim = 6

            # Extract action space
            act_space = self.gym_env.action_space
            if hasattr(act_space, "shape") and act_space.shape is not None:
                self.action_dim = int(act_space.shape[0])
                self.continuous = True
            elif hasattr(act_space, "n"):
                self.action_dim = int(act_space.n)
                self.continuous = False
            else:
                self.action_dim = 2
                self.continuous = True
        else:
            # Default specs for mock / uninstalled gym
            self.observation_dim = 6
            self.action_dim = 2
            self.continuous = True

        self.subgoal_dim = min(2, self.observation_dim)
        self.max_steps = max_steps or 200
        self.steps_taken = 0

    def reset(self, seed: Optional[int] = None) -> List[float]:
        self.steps_taken = 0
        if self.gym_env is not None:
            obs, info = self.gym_env.reset(seed=seed)
            return self._flatten_obs(obs)
        else:
            return [0.0] * self.observation_dim

    def _flatten_obs(self, obs: Any) -> List[float]:
        if isinstance(obs, dict):
            flat = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if hasattr(v, "tolist"):
                    flat.extend(v.tolist())
                elif isinstance(v, (list, tuple)):
                    flat.extend(list(v))
                else:
                    flat.append(float(v))
            return flat
        elif hasattr(obs, "tolist"):
            return obs.tolist()
        elif isinstance(obs, (list, tuple)):
            return list(obs)
        return [float(obs)]

    def step(self, action: Union[int, List[float]]) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        if self.gym_env is not None:
            import numpy as np
            if self.continuous and isinstance(action, list):
                act_arr = np.array(action, dtype=np.float32)
            else:
                act_arr = action

            next_obs, reward, terminated, truncated, info = self.gym_env.step(act_arr)
            done = bool(terminated or truncated or (self.steps_taken >= self.max_steps))
            flat_obs = self._flatten_obs(next_obs)
            return flat_obs, float(reward), done, info
        else:
            done = self.steps_taken >= self.max_steps
            return [0.0] * self.observation_dim, -0.01, done, {"is_success": False}
