import math
import random
from typing import Dict, Tuple, Any, List, Optional
from .vec_math import l2_dist, clip

class ContinuousGoalNavigationEnv:
    """
    Continuous 2D Multi-Room Physics Navigation Benchmark for Hierarchical Reinforcement Learning.
    Features:
    - Continuous State Space: [agent_x, agent_y, vel_x, vel_y, goal_x, goal_y] (dim = 6)
    - Continuous Action Space: 2D control force [a_x, a_y] in [-1.0, 1.0] (dim = 2)
    - Subgoal Dimension: 2D directional target [delta_x, delta_y] (dim = 2)
    - Second-order point mass physics with inertia, friction, and elastic wall boundaries.
    - Sparse goal rewards across bottlenecked multi-room layout.
    """
    def __init__(
        self,
        room_size: float = 10.0,
        max_steps: int = 200,
        dt: float = 0.2,
        friction: float = 0.08,
        goal_threshold: float = 0.8,
    ):
        self.room_size = room_size
        self.max_steps = max_steps
        self.dt = dt
        self.friction = friction
        self.goal_threshold = goal_threshold

        self.observation_dim = 6  # [x, y, vx, vy, gx, gy] normalized
        self.action_dim = 2       # [ax, ay] continuous control in [-1, 1]
        self.subgoal_dim = 2      # [delta_x, delta_y]
        self.continuous = True

        # State variables
        self.pos = [1.5, 1.5]
        self.vel = [0.0, 0.0]
        self.goal = [room_size - 1.5, room_size - 1.5]
        self.steps_taken = 0

        # Bottleneck partitions: 4 rooms with narrow doorways
        self.mid = self.room_size / 2.0
        self.door_width = 1.6

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            random.seed(seed)
        self.pos = [1.5, 1.5]
        self.vel = [0.0, 0.0]
        self.goal = [self.room_size - 1.5, self.room_size - 1.5]
        self.steps_taken = 0
        return self._get_obs()

    def _get_obs(self) -> List[float]:
        s = self.room_size
        return [
            self.pos[0] / s,
            self.pos[1] / s,
            self.vel[0] / 5.0,
            self.vel[1] / 5.0,
            self.goal[0] / s,
            self.goal[1] / s,
        ]

    def _collides_with_walls(self, x: float, y: float) -> bool:
        # Check outer boundaries
        if x < 0.2 or x > self.room_size - 0.2 or y < 0.2 or y > self.room_size - 0.2:
            return True

        # Check interior cross walls with doorways
        m = self.mid
        dw = self.door_width / 2.0

        # Vertical wall at x = mid
        if abs(x - m) < 0.3:
            # Doorway on vertical wall at y = mid / 2 and y = 3 * mid / 2
            door1 = abs(y - m * 0.5) < dw
            door2 = abs(y - m * 1.5) < dw
            if not (door1 or door2):
                return True

        # Horizontal wall at y = mid
        if abs(y - m) < 0.3:
            door3 = abs(x - m * 0.5) < dw
            door4 = abs(x - m * 1.5) < dw
            if not (door3 or door4):
                return True

        return False

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        self.steps_taken += 1

        # Action: [ax, ay] continuous force
        ax = clip(action[0], -1.0, 1.0)
        ay = clip(action[1], -1.0, 1.0)

        # Physics integration (Euler-Maruyama with friction)
        self.vel[0] = (1.0 - self.friction) * self.vel[0] + ax * self.dt * 4.0
        self.vel[1] = (1.0 - self.friction) * self.vel[1] + ay * self.dt * 4.0

        # Speed limit
        speed = math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)
        max_speed = 3.0
        if speed > max_speed:
            self.vel[0] = (self.vel[0] / speed) * max_speed
            self.vel[1] = (self.vel[1] / speed) * max_speed

        next_x = self.pos[0] + self.vel[0] * self.dt
        next_y = self.pos[1] + self.vel[1] * self.dt

        # Collision detection with walls
        if self._collides_with_walls(next_x, self.pos[1]):
            self.vel[0] = -0.3 * self.vel[0]  # Elastic bounce
        else:
            self.pos[0] = clip(next_x, 0.2, self.room_size - 0.2)

        if self._collides_with_walls(self.pos[0], next_y):
            self.vel[1] = -0.3 * self.vel[1]  # Elastic bounce
        else:
            self.pos[1] = clip(next_y, 0.2, self.room_size - 0.2)

        dist = l2_dist(self.pos, self.goal)
        is_success = dist < self.goal_threshold

        # Sparse extrinsic reward structure with minor control cost
        control_cost = 0.001 * (ax ** 2 + ay ** 2)
        if is_success:
            reward = 10.0 - control_cost
        else:
            reward = -0.01 - control_cost

        done = is_success or (self.steps_taken >= self.max_steps)

        info = {
            "is_success": is_success,
            "dist_to_goal": dist,
            "agent_pos": list(self.pos),
            "velocity": list(self.vel),
            "goal_pos": list(self.goal),
        }

        return self._get_obs(), reward, done, info
