import math
from typing import Tuple, Optional, Union, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    Normal = None
    Categorical = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    def get_default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    class TorchManagerNetwork(nn.Module):
        """
        High-Level Manager Policy (Macro Actor-Critic) in PyTorch.
        Processes state observation -> Emits latent directional sub-goal g_t and macro state-value V^M(s_t).
        """
        def __init__(
            self,
            obs_dim: int,
            subgoal_dim: int,
            hidden_dim: int = 64,
            use_layer_norm: bool = True,
        ):
            super().__init__()
            self.obs_dim = obs_dim
            self.subgoal_dim = subgoal_dim

            layers = [nn.Linear(obs_dim, hidden_dim)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

            self.shared_encoder = nn.Sequential(*layers)
            self.actor_head = nn.Linear(hidden_dim, subgoal_dim)
            self.critic_head = nn.Linear(hidden_dim, 1)

            # Subgoal log_std for continuous exploration
            self.log_std = nn.Parameter(torch.full((subgoal_dim,), -0.5))

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns (subgoal_mean, value_estimate)
            """
            feat = self.shared_encoder(obs)
            subgoal_mean = torch.tanh(self.actor_head(feat))
            value = self.critic_head(feat)
            return subgoal_mean, value

        def sample_subgoal(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (subgoal, log_prob, value)
            """
            mean, value = self.forward(obs)
            if deterministic:
                return mean, torch.zeros_like(mean[..., :1]), value

            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            raw_subgoal = dist.rsample()
            subgoal = torch.clamp(raw_subgoal, -1.0, 1.0)
            log_prob = dist.log_prob(raw_subgoal).sum(dim=-1, keepdim=True)
            return subgoal, log_prob, value

        def evaluate_subgoal(self, obs: torch.Tensor, subgoal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (log_prob, entropy, value)
            """
            mean, value = self.forward(obs)
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(subgoal).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return log_prob, entropy, value


    class TorchWorkerNetwork(nn.Module):
        """
        Low-Level Worker Policy (Micro Actor-Critic) in PyTorch.
        Conditioned on joint observation (s_t, g_t).
        Supports:
        - Continuous Control: Gaussian Actor (mean, log_std)
        - Discrete Control: Categorical Actor (action logits)
        """
        def __init__(
            self,
            obs_dim: int,
            subgoal_dim: int,
            action_dim: int,
            continuous: bool = False,
            hidden_dim: int = 64,
            use_layer_norm: bool = True,
        ):
            super().__init__()
            self.obs_dim = obs_dim
            self.subgoal_dim = subgoal_dim
            self.action_dim = action_dim
            self.continuous = continuous

            in_dim = obs_dim + subgoal_dim
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

            self.shared_encoder = nn.Sequential(*layers)

            if continuous:
                self.actor_mean = nn.Linear(hidden_dim, action_dim)
                self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
            else:
                self.actor_logits = nn.Linear(hidden_dim, action_dim)

            self.critic_head = nn.Linear(hidden_dim, 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            if self.continuous:
                nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            else:
                nn.init.orthogonal_(self.actor_logits.weight, gain=0.01)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

        def forward(self, obs: torch.Tensor, subgoal: torch.Tensor) -> Tuple[Any, torch.Tensor]:
            x = torch.cat([obs, subgoal], dim=-1)
            feat = self.shared_encoder(x)
            value = self.critic_head(feat)

            if self.continuous:
                mean = torch.tanh(self.actor_mean(feat))
                return mean, value
            else:
                logits = self.actor_logits(feat)
                return logits, value

        def sample_action(
            self,
            obs: torch.Tensor,
            subgoal: torch.Tensor,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (action, log_prob, value)
            """
            x = torch.cat([obs, subgoal], dim=-1)
            feat = self.shared_encoder(x)
            value = self.critic_head(feat)

            if self.continuous:
                mean = torch.tanh(self.actor_mean(feat))
                if deterministic:
                    return mean, torch.zeros_like(mean[..., :1]), value
                std = torch.exp(self.actor_log_std)
                dist = Normal(mean, std)
                raw_action = dist.rsample()
                action = torch.clamp(raw_action, -1.0, 1.0)
                log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
                return action, log_prob, value
            else:
                logits = self.actor_logits(feat)
                dist = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)
                return action, log_prob, value

        def evaluate_action(
            self,
            obs: torch.Tensor,
            subgoal: torch.Tensor,
            action: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (log_prob, entropy, value)
            """
            x = torch.cat([obs, subgoal], dim=-1)
            feat = self.shared_encoder(x)
            value = self.critic_head(feat)

            if self.continuous:
                mean = torch.tanh(self.actor_mean(feat))
                std = torch.exp(self.actor_log_std)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                return log_prob, entropy, value
            else:
                logits = self.actor_logits(feat)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(action.squeeze(-1) if action.dim() > 1 else action).unsqueeze(-1)
                entropy = dist.entropy().unsqueeze(-1)
                return log_prob, entropy, value
else:
    # Stubs when PyTorch is not available
    def get_default_device():
        return "cpu"
    TorchManagerNetwork = None
    TorchWorkerNetwork = None
