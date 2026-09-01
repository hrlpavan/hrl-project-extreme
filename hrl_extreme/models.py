import math
import random
from typing import Tuple, List, Optional, Union, Dict, Any
from .vec_math import (
    matmul_vec, relu, relu_grad, tanh, tanh_grad, softmax,
    gaussian_sample, gaussian_log_prob, clip, clip_vec
)

class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        std = math.sqrt(2.0 / in_features) if activation == "relu" else math.sqrt(1.0 / in_features)
        # Weights: [out_features, in_features]
        self.weights = [[random.gauss(0, std) for _ in range(in_features)] for _ in range(out_features)]
        self.biases = [0.0] * out_features
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features

        # Velocity for momentum optimizer
        self.v_weights = [[0.0] * in_features for _ in range(out_features)]
        self.v_biases = [0.0] * out_features

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        # Returns (activated_output, pre_activation)
        raw = matmul_vec(self.weights, x, self.biases)
        if self.activation == "relu":
            out = [relu(v) for v in raw]
        elif self.activation == "tanh":
            out = [tanh(v) for v in raw]
        elif self.activation == "softmax":
            out = softmax(raw)
        else:
            out = list(raw)
        return out, raw

    def backward_step(
        self,
        x: List[float],
        grad_output: List[float],
        out: List[float],
        lr: float = 0.001,
        momentum: float = 0.9,
    ) -> List[float]:
        """
        Backpropagates gradients and returns grad_input.
        """
        # Activation gradient
        if self.activation == "relu":
            d_raw = [g * relu_grad(v) for g, v in zip(grad_output, out)]
        elif self.activation == "tanh":
            d_raw = [g * tanh_grad(v) for g, v in zip(grad_output, out)]
        else:
            d_raw = list(grad_output)

        grad_input = [0.0] * self.in_features
        for i in range(self.out_features):
            d_i = d_raw[i]
            # Clip gradient to prevent explosion
            d_i = clip(d_i, -5.0, 5.0)
            self.v_biases[i] = momentum * self.v_biases[i] + lr * d_i
            self.biases[i] -= self.v_biases[i]
            for j in range(self.in_features):
                grad_input[j] += self.weights[i][j] * d_i
                grad_w = d_i * x[j]
                self.v_weights[i][j] = momentum * self.v_weights[i][j] + lr * grad_w
                self.weights[i][j] -= self.v_weights[i][j]

        return grad_input


class ManagerNetwork:
    """
    High-Level Policy (Macro Actor-Critic):
    Observes state -> Emits directional Sub-Goal g_t and macro state-value V^M(s_t).
    """
    def __init__(self, obs_dim: int, subgoal_dim: int, hidden_dim: int = 48):
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.hidden_dim = hidden_dim

        self.fc1 = DenseLayer(obs_dim, hidden_dim, activation="relu")
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, activation="relu")
        self.fc_actor = DenseLayer(hidden_dim, subgoal_dim, activation="tanh")
        self.fc_critic = DenseLayer(hidden_dim, 1, activation="linear")

        self.log_std = [-0.5] * subgoal_dim

    def forward(self, obs: List[float]) -> Tuple[List[float], float]:
        h1, _ = self.fc1.forward(obs)
        h2, _ = self.fc2.forward(h1)
        subgoal, _ = self.fc_actor.forward(h2)
        value, _ = self.fc_critic.forward(h2)
        return subgoal, value[0]

    def sample_subgoal(self, obs: List[float], deterministic: bool = False) -> Tuple[List[float], float, float]:
        subgoal_mean, value = self.forward(obs)
        if deterministic:
            return subgoal_mean, 0.0, value
        std = [math.exp(s) for s in self.log_std]
        raw_subgoal = gaussian_sample(subgoal_mean, std)
        subgoal = clip_vec(raw_subgoal, -1.0, 1.0)
        log_prob = gaussian_log_prob(raw_subgoal, subgoal_mean, std)
        return subgoal, log_prob, value

    def train_step(self, obs: List[float], target_subgoal: List[float], target_value: float, lr: float = 0.001) -> float:
        h1, _ = self.fc1.forward(obs)
        h2, _ = self.fc2.forward(h1)
        pred_subgoal, _ = self.fc_actor.forward(h2)
        pred_val, _ = self.fc_critic.forward(h2)

        v_error = pred_val[0] - target_value
        critic_loss = v_error ** 2

        # Actor error
        actor_grad = [2.0 * (p - t) for p, t in zip(pred_subgoal, target_subgoal)]
        critic_grad = [2.0 * v_error]

        # Backpropagation
        d_h2_actor = self.fc_actor.backward_step(h2, actor_grad, pred_subgoal, lr=lr)
        d_h2_critic = self.fc_critic.backward_step(h2, critic_grad, pred_val, lr=lr)
        d_h2 = [a + c for a, c in zip(d_h2_actor, d_h2_critic)]

        d_h1 = self.fc2.backward_step(h1, d_h2, h2, lr=lr)
        self.fc1.backward_step(obs, d_h1, h1, lr=lr)

        return float(critic_loss)


class WorkerNetwork:
    """
    Low-Level Policy (Micro Actor-Critic):
    Conditioned on (obs, subgoal).
    Supports:
    - Continuous control: Gaussian actor emitting continuous action vectors in [-1.0, 1.0].
    - Discrete control: Softmax actor emitting categorical action probabilities.
    """
    def __init__(
        self,
        obs_dim: int,
        subgoal_dim: int,
        action_dim: int,
        continuous: bool = False,
        hidden_dim: int = 48,
    ):
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.hidden_dim = hidden_dim

        in_dim = obs_dim + subgoal_dim
        self.fc1 = DenseLayer(in_dim, hidden_dim, activation="relu")
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, activation="relu")

        if continuous:
            self.fc_actor = DenseLayer(hidden_dim, action_dim, activation="tanh")
            self.log_std = [-0.5] * action_dim
        else:
            self.fc_actor = DenseLayer(hidden_dim, action_dim, activation="softmax")
            self.log_std = []

        self.fc_critic = DenseLayer(hidden_dim, 1, activation="linear")

    def forward(self, obs: List[float], subgoal: List[float]) -> Tuple[Union[List[float], float], float]:
        x = obs + subgoal
        h1, _ = self.fc1.forward(x)
        h2, _ = self.fc2.forward(h1)
        action_out, _ = self.fc_actor.forward(h2)
        val, _ = self.fc_critic.forward(h2)
        return action_out, val[0]

    def sample_action(
        self,
        obs: List[float],
        subgoal: List[float],
        deterministic: bool = False,
    ) -> Tuple[Union[int, List[float]], float, float]:
        """
        Returns (action, log_prob, value)
        """
        x = obs + subgoal
        h1, _ = self.fc1.forward(x)
        h2, _ = self.fc2.forward(h1)
        actor_out, _ = self.fc_actor.forward(h2)
        val, _ = self.fc_critic.forward(h2)
        value = val[0]

        if self.continuous:
            mean = actor_out
            if deterministic:
                return mean, 0.0, value
            std = [math.exp(s) for s in self.log_std]
            raw_action = gaussian_sample(mean, std)
            action = clip_vec(raw_action, -1.0, 1.0)
            log_prob = gaussian_log_prob(raw_action, mean, std)
            return action, log_prob, value
        else:
            action_probs = actor_out
            if deterministic:
                action = int(max(range(len(action_probs)), key=lambda i: action_probs[i]))
                log_prob = math.log(max(1e-12, action_probs[action]))
                return action, log_prob, value
            r = random.random()
            cum = 0.0
            action = len(action_probs) - 1
            for i, p in enumerate(action_probs):
                cum += p
                if r <= cum:
                    action = i
                    break
            log_prob = math.log(max(1e-12, action_probs[action]))
            return action, log_prob, value

    def train_step(
        self,
        obs: List[float],
        subgoal: List[float],
        action: Union[int, List[float]],
        advantage: float,
        target_value: float,
        lr: float = 0.001,
    ) -> float:
        """
        Policy gradient + value regression step.
        """
        x = obs + subgoal
        h1, _ = self.fc1.forward(x)
        h2, _ = self.fc2.forward(h1)
        actor_out, _ = self.fc_actor.forward(h2)
        pred_val, _ = self.fc_critic.forward(h2)

        v_error = pred_val[0] - target_value
        critic_loss = v_error ** 2
        critic_grad = [2.0 * v_error]

        if self.continuous:
            # Policy gradient for continuous Gaussian: -advantage * (action - mean) / var
            mean = actor_out
            std = [math.exp(s) for s in self.log_std]
            actor_grad = [
                -advantage * (a - m) / (s ** 2)
                for a, m, s in zip(action, mean, std)
            ]
        else:
            # Policy gradient for discrete categorical: -advantage * grad(log_prob)
            probs = actor_out
            actor_grad = [0.0] * self.action_dim
            for i in range(self.action_dim):
                if i == action:
                    actor_grad[i] = -advantage * (1.0 - probs[i])
                else:
                    actor_grad[i] = -advantage * (-probs[i])

        d_h2_actor = self.fc_actor.backward_step(h2, actor_grad, actor_out, lr=lr)
        d_h2_critic = self.fc_critic.backward_step(h2, critic_grad, pred_val, lr=lr)
        d_h2 = [a + c for a, c in zip(d_h2_actor, d_h2_critic)]

        d_h1 = self.fc2.backward_step(h1, d_h2, h2, lr=lr)
        self.fc1.backward_step(x, d_h1, h1, lr=lr)

        return float(critic_loss)
