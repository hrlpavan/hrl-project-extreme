import random
from typing import Tuple, Dict, Any, List
from .models import ManagerNetwork, WorkerNetwork
from .buffer import HierarchicalReplayBuffer
from .vec_math import cosine_similarity, l2_dist, sub

class HierarchicalAgent:
    """
    FeUdal Hierarchical Reinforcement Learning Agent.
    - Manager: Sets directional sub-goals every `c_step` interval.
    - Worker: Executes primitive actions to satisfy the manager's sub-goal.
    - Intrinsic Motivation: Cosine & distance similarity in state transition space.
    """
    def __init__(
        self,
        obs_dim: int = 4,
        subgoal_dim: int = 2,
        action_dim: int = 4,
        c_step: int = 8,
        gamma_manager: float = 0.99,
        gamma_worker: float = 0.95,
        learning_rate: float = 0.001,
    ):
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.action_dim = action_dim
        self.c_step = c_step
        self.gamma_m = gamma_manager
        self.gamma_w = gamma_worker
        self.lr = learning_rate

        self.manager = ManagerNetwork(obs_dim, subgoal_dim)
        self.worker = WorkerNetwork(obs_dim, subgoal_dim, action_dim)
        self.buffer = HierarchicalReplayBuffer()

        # Step tracking
        self.step_in_subgoal = 0
        self.current_subgoal = [0.0] * subgoal_dim
        self.manager_start_obs = None
        self.cumulative_extrinsic_reward = 0.0

    def compute_intrinsic_reward(self, obs: List[float], next_obs: List[float], subgoal: List[float]) -> float:
        state_delta = sub(next_obs[:2], obs[:2])
        dist = l2_dist(state_delta, subgoal)
        cos_sim = cosine_similarity(state_delta, subgoal)
        return float(cos_sim - dist)

    def select_action(self, obs: List[float], evaluate: bool = False) -> Tuple[int, List[float]]:
        # If at subgoal boundary, generate a new macro subgoal
        if self.step_in_subgoal % self.c_step == 0 or self.manager_start_obs is None:
            self.manager_start_obs = list(obs)
            subgoal, _ = self.manager.forward(obs)
            if not evaluate:
                # Add exploratory noise
                subgoal = [max(-1.0, min(1.0, g + random.gauss(0, 0.1))) for g in subgoal]
            self.current_subgoal = subgoal
            self.cumulative_extrinsic_reward = 0.0
            self.step_in_subgoal = 0

        # Worker generates primitive action
        action_probs, _ = self.worker.forward(obs, self.current_subgoal)

        if evaluate:
            action = int(max(range(len(action_probs)), key=lambda i: action_probs[i]))
        else:
            r = random.random()
            cum = 0.0
            action = len(action_probs) - 1
            for i, p in enumerate(action_probs):
                cum += p
                if r <= cum:
                    action = i
                    break

        self.step_in_subgoal += 1
        return action, self.current_subgoal

    def step_update(
        self,
        obs: List[float],
        action: int,
        extrinsic_reward: float,
        next_obs: List[float],
        done: bool,
    ):
        intrinsic_reward = self.compute_intrinsic_reward(obs, next_obs, self.current_subgoal)
        self.cumulative_extrinsic_reward += extrinsic_reward

        # 1. Store worker transition
        self.buffer.store_worker(
            obs=obs,
            subgoal=self.current_subgoal,
            action=action,
            intrinsic_reward=intrinsic_reward,
            next_obs=next_obs,
            done=done,
        )

        # 2. Store manager transition at macro boundary
        if self.step_in_subgoal >= self.c_step or done:
            self.buffer.store_manager(
                obs=self.manager_start_obs,
                subgoal=self.current_subgoal,
                cumulative_extrinsic_reward=self.cumulative_extrinsic_reward,
                next_obs=next_obs,
                done=done,
            )
            self.manager_start_obs = list(next_obs)
            self.cumulative_extrinsic_reward = 0.0
            self.step_in_subgoal = 0

    def train_batch(self, batch_size: int = 32) -> Dict[str, float]:
        w_batch = self.buffer.sample_worker(batch_size)
        m_batch = self.buffer.sample_manager(batch_size)

        w_loss = 0.0
        m_loss = 0.0

        if w_batch:
            for item in w_batch:
                _, v_curr = self.worker.forward(item["obs"], item["subgoal"])
                _, v_next = self.worker.forward(item["next_obs"], item["subgoal"])
                target = item["reward"] + (0.0 if item["done"] else self.gamma_w * v_next)
                td_error = target - v_curr
                w_loss += td_error ** 2
            w_loss /= len(w_batch)

        if m_batch:
            for item in m_batch:
                _, v_curr = self.manager.forward(item["obs"])
                _, v_next = self.manager.forward(item["next_obs"])
                target = item["reward"] + (0.0 if item["done"] else self.gamma_m * v_next)
                td_error = target - v_curr
                m_loss += td_error ** 2
            m_loss /= len(m_batch)

        return {"worker_loss": float(w_loss), "manager_loss": float(m_loss)}
