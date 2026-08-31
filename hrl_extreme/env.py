import math
import random
from typing import Dict, Tuple, Any, List, Optional
from .vec_math import l2_dist

class SparseGoalMazeEnv:
    """
    Long-Horizon Sparse-Reward Multi-Room Maze Environment.
    Standard flat RL fails here due to sparse reward horizons (>50 steps),
    making it the ideal benchmark for Hierarchical Reinforcement Learning.
    """
    def __init__(self, grid_size: int = 12, max_steps: int = 150):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_pos = [1.0, 1.0]
        self.goal_pos = [float(grid_size - 2), float(grid_size - 2)]
        self.steps_taken = 0

        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space_n = 4
        self.observation_dim = 4  # [agent_x, agent_y, goal_x, goal_y] normalized
        self.subgoal_dim = 2       # [delta_x, delta_y]

        # Inner walls separating rooms with narrow bottlenecks
        self.walls = set()
        mid = grid_size // 2
        for i in range(grid_size):
            if i != 2 and i != grid_size - 3:
                self.walls.add((mid, i))
                self.walls.add((i, mid))

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            random.seed(seed)
        self.agent_pos = [1.0, 1.0]
        self.goal_pos = [float(self.grid_size - 2), float(self.grid_size - 2)]
        self.steps_taken = 0
        return self._get_obs()

    def _get_obs(self) -> List[float]:
        return [
            self.agent_pos[0] / self.grid_size,
            self.agent_pos[1] / self.grid_size,
            self.goal_pos[0] / self.grid_size,
            self.goal_pos[1] / self.grid_size,
        ]

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        delta = [0.0, 0.0]

        if action == 0:    # Up
            delta = [0.0, 1.0]
        elif action == 1:  # Down
            delta = [0.0, -1.0]
        elif action == 2:  # Left
            delta = [-1.0, 0.0]
        elif action == 3:  # Right
            delta = [1.0, 0.0]

        next_x = self.agent_pos[0] + delta[0]
        next_y = self.agent_pos[1] + delta[1]

        # Check boundaries and walls
        if (0 <= next_x < self.grid_size and
            0 <= next_y < self.grid_size and
            (int(next_x), int(next_y)) not in self.walls):
            self.agent_pos = [next_x, next_y]

        dist = l2_dist(self.agent_pos, self.goal_pos)
        is_success = dist < 0.5

        # Sparse extrinsic reward: +10 on reaching goal, -0.01 step penalty
        reward = 10.0 if is_success else -0.01
        done = is_success or (self.steps_taken >= self.max_steps)

        info = {
            "is_success": is_success,
            "dist_to_goal": dist,
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
        }

        return self._get_obs(), reward, done, info
