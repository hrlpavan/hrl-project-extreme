import math
import random
from typing import Tuple, Dict, Any, List, Optional, Union
from .models import ManagerNetwork as NativeManager, WorkerNetwork as NativeWorker
from .buffer import HierarchicalReplayBuffer
from .vec_math import cosine_similarity, l2_dist, sub, clip_vec
from .reachability import ReachabilityProjector

# Conditional PyTorch import
try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from .torch_models import TorchManagerNetwork, TorchWorkerNetwork, get_default_device
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class HierarchicalAgent:
    """
    Production-Grade FeUdal Hierarchical Reinforcement Learning Agent.
    - High-Level Manager: Formulates directional sub-goals in state space every `c_step`.
    - Low-Level Worker: Executes continuous / discrete primitives conditioned on sub-goals.
    - Reachability Analysis: GARA/STAR set-based forward reachability projection R_c(s).
    - Intrinsic Motivation: Cosine alignment and Euclidean state transition reward.
    - Dual Engine: Seamless PyTorch Deep Policy Optimization or Native Vectorized Engine.
    """
    def __init__(
        self,
        obs_dim: int = 6,
        subgoal_dim: int = 2,
        action_dim: int = 2,
        continuous: bool = True,
        c_step: int = 8,
        gamma_manager: float = 0.99,
        gamma_worker: float = 0.95,
        learning_rate: float = 0.001,
        use_torch: bool = True,
        use_reachability: bool = True,
    ):
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.c_step = c_step
        self.gamma_m = gamma_manager
        self.gamma_w = gamma_worker
        self.lr = learning_rate
        self.use_reachability = use_reachability

        self.reachability_projector = ReachabilityProjector(c_step=c_step, room_size=10.0)
        self.reachability_feasibility = 1.0
        self.reachability_debug = {}

        self.use_torch = use_torch and TORCH_AVAILABLE
        self.device = get_default_device() if (self.use_torch and TORCH_AVAILABLE) else "cpu"

        if self.use_torch:
            self.manager = TorchManagerNetwork(obs_dim, subgoal_dim).to(self.device)
            self.worker = TorchWorkerNetwork(obs_dim, subgoal_dim, action_dim, continuous=continuous).to(self.device)
            self.optimizer_m = optim.Adam(self.manager.parameters(), lr=learning_rate)
            self.optimizer_w = optim.Adam(self.worker.parameters(), lr=learning_rate)
        else:
            self.manager = NativeManager(obs_dim, subgoal_dim)
            self.worker = NativeWorker(obs_dim, subgoal_dim, action_dim, continuous=continuous)

        self.buffer = HierarchicalReplayBuffer()

        # Step tracking for single-env rollouts
        self.step_in_subgoal = 0
        self.current_subgoal = [0.0] * subgoal_dim
        self.manager_start_obs = None
        self.cumulative_extrinsic_reward = 0.0

    def compute_intrinsic_reward(self, obs: List[float], next_obs: List[float], subgoal: List[float]) -> float:
        # Spatial change in top-2 coordinate dimensions
        state_delta = sub(next_obs[:2], obs[:2])
        dist = l2_dist(state_delta, subgoal)
        cos_sim = cosine_similarity(state_delta, subgoal)
        return float(cos_sim - 0.5 * dist)

    def select_action(
        self,
        obs: List[float],
        evaluate: bool = False,
        goal_pos: Optional[List[float]] = None,
    ) -> Tuple[Union[int, List[float]], List[float], float, float]:
        """
        Select action for single-environment step with reachability guidance.
        Returns: (action, current_subgoal, log_prob, value)
        """
        # Manager sub-goal generation at macro boundary
        if self.step_in_subgoal % self.c_step == 0 or self.manager_start_obs is None:
            self.manager_start_obs = list(obs)
            if self.use_torch:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    sg_t, _, _ = self.manager.sample_subgoal(obs_t, deterministic=evaluate)
                    subgoal = sg_t.squeeze(0).cpu().tolist()
            else:
                subgoal, _, _ = self.manager.sample_subgoal(obs, deterministic=evaluate)

            if self.use_reachability:
                target_g = goal_pos if goal_pos else (obs[2:4] if len(obs) >= 4 else None)
                vel = obs[4:6] if len(obs) >= 6 else None
                subgoal, feas, r_debug = self.reachability_projector.project_subgoal(
                    agent_pos=obs[:2],
                    raw_subgoal=subgoal,
                    goal_pos=target_g,
                    velocity=vel
                )
                self.reachability_feasibility = feas
                self.reachability_debug = r_debug

            self.current_subgoal = subgoal
            self.cumulative_extrinsic_reward = 0.0
            self.step_in_subgoal = 0

        # Worker primitive action generation
        if self.use_torch:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                sg_t = torch.tensor(self.current_subgoal, dtype=torch.float32, device=self.device).unsqueeze(0)
                act_t, lp_t, val_t = self.worker.sample_action(obs_t, sg_t, deterministic=evaluate)

                if self.continuous:
                    action = act_t.squeeze(0).cpu().tolist()
                else:
                    action = int(act_t.squeeze().cpu().item())
                log_prob = float(lp_t.squeeze().cpu().item())
                value = float(val_t.squeeze().cpu().item())
        else:
            action, log_prob, value = self.worker.sample_action(obs, self.current_subgoal, deterministic=evaluate)

        self.step_in_subgoal += 1
        return action, list(self.current_subgoal), log_prob, value

    def select_actions_vec(
        self,
        obs_list: List[List[float]],
        steps_in_subgoal: List[int],
        current_subgoals: List[List[float]],
        evaluate: bool = False,
    ) -> Tuple[List[Union[int, List[float]]], List[List[float]], List[float], List[float]]:
        """
        Batched action selection for parallel VectorEnv workers.
        """
        num_envs = len(obs_list)
        updated_subgoals = [list(sg) for sg in current_subgoals]

        # Determine which envs need new subgoals
        need_subgoal_idx = [i for i in range(num_envs) if steps_in_subgoal[i] % self.c_step == 0]

        if need_subgoal_idx:
            if self.use_torch:
                sg_obs = [obs_list[i] for i in need_subgoal_idx]
                with torch.no_grad():
                    obs_t = torch.tensor(sg_obs, dtype=torch.float32, device=self.device)
                    sg_t, _, _ = self.manager.sample_subgoal(obs_t, deterministic=evaluate)
                    new_sgs = sg_t.cpu().tolist()
                for idx, sg in zip(need_subgoal_idx, new_sgs):
                    updated_subgoals[idx] = sg
            else:
                for idx in need_subgoal_idx:
                    sg, _, _ = self.manager.sample_subgoal(obs_list[idx], deterministic=evaluate)
                    updated_subgoals[idx] = sg

        # Sample actions across all envs in batch
        if self.use_torch:
            with torch.no_grad():
                obs_t = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
                sg_t = torch.tensor(updated_subgoals, dtype=torch.float32, device=self.device)
                act_t, lp_t, val_t = self.worker.sample_action(obs_t, sg_t, deterministic=evaluate)

                if self.continuous:
                    actions = act_t.cpu().tolist()
                else:
                    actions = [int(a) for a in act_t.view(-1).cpu().tolist()]
                log_probs = [float(lp) for lp in lp_t.view(-1).cpu().tolist()]
                values = [float(v) for v in val_t.view(-1).cpu().tolist()]
        else:
            actions = []
            log_probs = []
            values = []
            for i in range(num_envs):
                act, lp, val = self.worker.sample_action(obs_list[i], updated_subgoals[i], deterministic=evaluate)
                actions.append(act)
                log_probs.append(lp)
                values.append(val)

        return actions, updated_subgoals, log_probs, values

    def step_update(
        self,
        obs: List[float],
        action: Union[int, List[float]],
        extrinsic_reward: float,
        next_obs: List[float],
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        intrinsic_reward = self.compute_intrinsic_reward(obs, next_obs, self.current_subgoal)
        self.cumulative_extrinsic_reward += extrinsic_reward

        # 1. Store worker micro-transition
        self.buffer.store_worker(
            obs=obs,
            subgoal=self.current_subgoal,
            action=action,
            intrinsic_reward=intrinsic_reward,
            next_obs=next_obs,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        # 2. Store manager macro-transition at boundary
        if self.step_in_subgoal >= self.c_step or done:
            self.buffer.store_manager(
                obs=self.manager_start_obs,
                subgoal=self.current_subgoal,
                cumulative_extrinsic_reward=self.cumulative_extrinsic_reward,
                next_obs=next_obs,
                done=done,
                log_prob=0.0,
                value=0.0,
            )
            self.manager_start_obs = list(next_obs)
            self.cumulative_extrinsic_reward = 0.0
            self.step_in_subgoal = 0

    def train_batch(self, batch_size: int = 32) -> Dict[str, float]:
        w_batch = self.buffer.sample_worker(batch_size)
        m_batch = self.buffer.sample_manager(batch_size)

        if not w_batch and not m_batch:
            return {"worker_loss": 0.0, "manager_loss": 0.0}

        if self.use_torch:
            return self._train_torch(w_batch, m_batch)
        else:
            return self._train_native(w_batch, m_batch)

    def _train_torch(self, w_batch: Optional[List[Dict[str, Any]]], m_batch: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        w_loss_val = 0.0
        m_loss_val = 0.0

        if w_batch:
            obs = torch.tensor([b["obs"] for b in w_batch], dtype=torch.float32, device=self.device)
            subgoals = torch.tensor([b["subgoal"] for b in w_batch], dtype=torch.float32, device=self.device)
            rewards = torch.tensor([b["reward"] for b in w_batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_obs = torch.tensor([b["next_obs"] for b in w_batch], dtype=torch.float32, device=self.device)
            dones = torch.tensor([float(b["done"]) for b in w_batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

            if self.continuous:
                actions = torch.tensor([b["action"] for b in w_batch], dtype=torch.float32, device=self.device)
            else:
                actions = torch.tensor([b["action"] for b in w_batch], dtype=torch.long, device=self.device)

            with torch.no_grad():
                _, next_vals = self.worker(next_obs, subgoals)
                target_vals = rewards + (1.0 - dones) * self.gamma_w * next_vals

            log_probs, entropy, values = self.worker.evaluate_action(obs, subgoals, actions)
            advantages = (target_vals - values).detach()

            # Actor loss + Value loss + Entropy regularization
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, target_vals)
            entropy_loss = -0.01 * entropy.mean()
            w_total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optimizer_w.zero_grad()
            w_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=1.0)
            self.optimizer_w.step()
            w_loss_val = float(w_total_loss.item())

        if m_batch:
            m_obs = torch.tensor([b["obs"] for b in m_batch], dtype=torch.float32, device=self.device)
            m_subgoals = torch.tensor([b["subgoal"] for b in m_batch], dtype=torch.float32, device=self.device)
            m_rewards = torch.tensor([b["reward"] for b in m_batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            m_next_obs = torch.tensor([b["next_obs"] for b in m_batch], dtype=torch.float32, device=self.device)
            m_dones = torch.tensor([float(b["done"]) for b in m_batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                _, next_m_vals = self.manager(m_next_obs)
                target_m_vals = m_rewards + (1.0 - m_dones) * self.gamma_m * next_m_vals

            m_log_probs, m_entropy, m_values = self.manager.evaluate_subgoal(m_obs, m_subgoals)
            m_advantages = (target_m_vals - m_values).detach()

            m_actor_loss = -(m_log_probs * m_advantages).mean()
            m_critic_loss = F.mse_loss(m_values, target_m_vals)
            m_total_loss = m_actor_loss + 0.5 * m_critic_loss

            self.optimizer_m.zero_grad()
            m_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), max_norm=1.0)
            self.optimizer_m.step()
            m_loss_val = float(m_total_loss.item())

        return {"worker_loss": w_loss_val, "manager_loss": m_loss_val}

    def _train_native(self, w_batch: Optional[List[Dict[str, Any]]], m_batch: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        w_loss = 0.0
        m_loss = 0.0

        if w_batch:
            for item in w_batch:
                _, v_next = self.worker.forward(item["next_obs"], item["subgoal"])
                target = item["reward"] + (0.0 if item["done"] else self.gamma_w * v_next)
                _, v_curr = self.worker.forward(item["obs"], item["subgoal"])
                advantage = target - v_curr
                step_loss = self.worker.train_step(
                    obs=item["obs"],
                    subgoal=item["subgoal"],
                    action=item["action"],
                    advantage=advantage,
                    target_value=target,
                    lr=self.lr,
                )
                w_loss += step_loss
            w_loss /= len(w_batch)

        if m_batch:
            for item in m_batch:
                _, v_next = self.manager.forward(item["next_obs"])
                target = item["reward"] + (0.0 if item["done"] else self.gamma_m * v_next)
                step_loss = self.manager.train_step(
                    obs=item["obs"],
                    target_subgoal=item["subgoal"],
                    target_value=target,
                    lr=self.lr,
                )
                m_loss += step_loss
            m_loss /= len(m_batch)

        return {"worker_loss": float(w_loss), "manager_loss": float(m_loss)}
