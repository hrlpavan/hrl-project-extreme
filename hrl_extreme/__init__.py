"""
HRL Project Extreme: Hierarchical Reinforcement Learning Engine
Featuring PyTorch Deep Policy Optimization, FeUdal Networks (FuN), Continuous Control, and Parallel Vectorized Workers.
"""

from .agent import HierarchicalAgent
from .buffer import HierarchicalReplayBuffer
from .env import SparseGoalMazeEnv
from .continuous_env import ContinuousGoalNavigationEnv
from .gym_wrapper import GymEnvWrapper
from .vec_env import VectorEnv, make_vec_env
from .models import ManagerNetwork, WorkerNetwork

try:
    from .torch_models import TorchManagerNetwork, TorchWorkerNetwork, TORCH_AVAILABLE
except ImportError:
    TorchManagerNetwork = None
    TorchWorkerNetwork = None
    TORCH_AVAILABLE = False

__all__ = [
    "HierarchicalAgent",
    "HierarchicalReplayBuffer",
    "SparseGoalMazeEnv",
    "ContinuousGoalNavigationEnv",
    "GymEnvWrapper",
    "VectorEnv",
    "make_vec_env",
    "ManagerNetwork",
    "WorkerNetwork",
    "TorchManagerNetwork",
    "TorchWorkerNetwork",
    "TORCH_AVAILABLE",
]

__version__ = "2.0.0"
