import math
import random
from typing import Tuple, List
from .vec_math import matmul_vec, relu, tanh, softmax

class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        std = math.sqrt(2.0 / in_features) if activation == "relu" else math.sqrt(1.0 / in_features)
        # Weights: [out_features, in_features]
        self.weights = [[random.gauss(0, std) for _ in range(in_features)] for _ in range(out_features)]
        self.biases = [0.0] * out_features
        self.activation = activation

    def forward(self, x: List[float]) -> List[float]:
        raw = matmul_vec(self.weights, x, self.biases)
        if self.activation == "relu":
            return [relu(v) for v in raw]
        elif self.activation == "tanh":
            return [tanh(v) for v in raw]
        elif self.activation == "softmax":
            return softmax(raw)
        return raw

class ManagerNetwork:
    """
    High-Level Policy: Observes full state -> Emits directional Sub-Goal in latent space.
    """
    def __init__(self, obs_dim: int, subgoal_dim: int, hidden_dim: int = 32):
        self.obs_dim = obs_dim
        self.subgoal_dim = subgoal_dim
        self.fc1 = DenseLayer(obs_dim, hidden_dim, activation="relu")
        self.fc_actor = DenseLayer(hidden_dim, subgoal_dim, activation="tanh")
        self.fc_critic = DenseLayer(hidden_dim, 1, activation="linear")

    def forward(self, obs: List[float]) -> Tuple[List[float], float]:
        h = self.fc1.forward(obs)
        subgoal = self.fc_actor.forward(h)
        value = self.fc_critic.forward(h)[0]
        return subgoal, value

class WorkerNetwork:
    """
    Low-Level Policy: Conditioned on state + Sub-Goal -> Emits primitive action probabilities.
    """
    def __init__(self, obs_dim: int, subgoal_dim: int, action_dim: int, hidden_dim: int = 32):
        in_dim = obs_dim + subgoal_dim
        self.fc1 = DenseLayer(in_dim, hidden_dim, activation="relu")
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, activation="relu")
        self.fc_actor = DenseLayer(hidden_dim, action_dim, activation="softmax")
        self.fc_critic = DenseLayer(hidden_dim, 1, activation="linear")

    def forward(self, obs: List[float], subgoal: List[float]) -> Tuple[List[float], float]:
        x = obs + subgoal
        h1 = self.fc1.forward(x)
        h2 = self.fc2.forward(h1)
        action_probs = self.fc_actor.forward(h2)
        value = self.fc_critic.forward(h2)[0]
        return action_probs, value
