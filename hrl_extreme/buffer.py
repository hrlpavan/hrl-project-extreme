import random
from typing import Dict, List, Any, Optional, Tuple, Union
from .vec_math import compute_gae

class HierarchicalReplayBuffer:
    """
    Dual-Timescale Hierarchical Replay Buffer with Generalized Advantage Estimation (GAE):
    - Worker Buffer: Stores single-step micro-transitions (s_t, g_t, a_t, r_intrinsic, s_{t+1}, done, log_prob, value)
    - Manager Buffer: Stores macro-step transitions (s_start, g_t, r_extrinsic_cumulative, s_{t+c}, done, log_prob, value)
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.worker_buffer: List[Dict[str, Any]] = []
        self.manager_buffer: List[Dict[str, Any]] = []

    def store_worker(
        self,
        obs: List[float],
        subgoal: List[float],
        action: Union[int, List[float]],
        intrinsic_reward: float,
        next_obs: List[float],
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        if len(self.worker_buffer) >= self.capacity:
            self.worker_buffer.pop(0)
        self.worker_buffer.append({
            "obs": list(obs),
            "subgoal": list(subgoal),
            "action": list(action) if isinstance(action, (list, tuple)) else action,
            "reward": float(intrinsic_reward),
            "next_obs": list(next_obs),
            "done": bool(done),
            "log_prob": float(log_prob),
            "value": float(value),
        })

    def store_manager(
        self,
        obs: List[float],
        subgoal: List[float],
        cumulative_extrinsic_reward: float,
        next_obs: List[float],
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        if len(self.manager_buffer) >= self.capacity:
            self.manager_buffer.pop(0)
        self.manager_buffer.append({
            "obs": list(obs),
            "subgoal": list(subgoal),
            "reward": float(cumulative_extrinsic_reward),
            "next_obs": list(next_obs),
            "done": bool(done),
            "log_prob": float(log_prob),
            "value": float(value),
        })

    def sample_worker(self, batch_size: int = 32) -> Optional[List[Dict[str, Any]]]:
        if len(self.worker_buffer) < batch_size:
            return None
        return random.sample(self.worker_buffer, batch_size)

    def sample_manager(self, batch_size: int = 32) -> Optional[List[Dict[str, Any]]]:
        if len(self.manager_buffer) < batch_size:
            return None
        return random.sample(self.manager_buffer, batch_size)

    def clear(self):
        self.worker_buffer.clear()
        self.manager_buffer.clear()

    def get_recent_worker_gae(self, n_steps: int = 64, gamma: float = 0.95, lam: float = 0.95) -> List[Dict[str, Any]]:
        """
        Computes GAE on recent sequential worker transitions.
        """
        recent = self.worker_buffer[-n_steps:]
        if not recent:
            return []
        rewards = [item["reward"] for item in recent]
        values = [item["value"] for item in recent] + [0.0]
        dones = [item["done"] for item in recent]
        advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        for i, item in enumerate(recent):
            item["advantage"] = advantages[i]
            item["return"] = returns[i]
        return recent

    def __len__(self) -> int:
        return len(self.worker_buffer)
