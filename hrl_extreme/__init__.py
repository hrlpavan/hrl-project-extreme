"""
HRL Project Extreme: Hierarchical Reinforcement Learning Engine
Featuring FeUdal Networks (FuN) and Goal-Conditioned Hierarchical Policy Optimization.
"""

from .agent import HierarchicalAgent
from .buffer import HierarchicalReplayBuffer
from .env import SparseGoalMazeEnv
from .models import ManagerNetwork, WorkerNetwork

__all__ = [
    "HierarchicalAgent",
    "HierarchicalReplayBuffer",
    "SparseGoalMazeEnv",
    "ManagerNetwork",
    "WorkerNetwork",
]

__version__ = "1.0.0"
