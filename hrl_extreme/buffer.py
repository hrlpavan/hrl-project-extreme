import random
from typing import Dict, List, Any, Optional

class HierarchicalReplayBuffer:
    """
    Dual-Timescale Hierarchical Replay Buffer:
    - Worker Buffer: Stores (s_t, g_t, a_t, r_intrinsic, s_{t+1}, done)
    - Manager Buffer: Stores (s_t, g_t, r_extrinsic_cumulative, s_{t+c}, done)
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.worker_buffer: List[Dict[str, Any]] = []
        self.manager_buffer: List[Dict[str, Any]] = []

    def store_worker(self, obs: List[float], subgoal: List[float], action: int,
                     intrinsic_reward: float, next_obs: List[float], done: bool):
        if len(self.worker_buffer) >= self.capacity:
            self.worker_buffer.pop(0)
        self.worker_buffer.append({
            "obs": list(obs),
            "subgoal": list(subgoal),
            "action": action,
            "reward": intrinsic_reward,
            "next_obs": list(next_obs),
            "done": done,
        })

    def store_manager(self, obs: List[float], subgoal: List[float],
                      cumulative_extrinsic_reward: float, next_obs: List[float], done: bool):
        if len(self.manager_buffer) >= self.capacity:
            self.manager_buffer.pop(0)
        self.manager_buffer.append({
            "obs": list(obs),
            "subgoal": list(subgoal),
            "reward": cumulative_extrinsic_reward,
            "next_obs": list(next_obs),
            "done": done,
        })

    def sample_worker(self, batch_size: int = 32) -> Optional[List[Dict[str, Any]]]:
        if len(self.worker_buffer) < batch_size:
            return None
        return random.sample(self.worker_buffer, batch_size)

    def sample_manager(self, batch_size: int = 32) -> Optional[List[Dict[str, Any]]]:
        if len(self.manager_buffer) < batch_size:
            return None
        return random.sample(self.manager_buffer, batch_size)

    def __len__(self) -> int:
        return len(self.worker_buffer)
