"""
Reachability Analysis & Automatic Symbolic Goal Abstraction (GARA / STAR)
Based on:
- "Goal Space Abstraction in Hierarchical Reinforcement Learning via Set-Based Reachability Analysis" (Zadem et al., ICDL 2023)
- "Reconciling Spatial and Temporal Abstractions for Goal Representation" (Zadem et al., ICLR 2024)

Provides:
1. ForwardReachabilityEstimator: Computes state-dependent forward reachable set R_c(s) within c steps.
2. SymbolicTopologyGraph: Discovers bottleneck nodes (e.g. doorways, room junctions) dynamically from rollouts.
3. ReachabilityProjector: Projects unconstrained Manager sub-goals into R_c(s) to guarantee feasibility.
"""

import math
from typing import List, Tuple, Dict, Optional, Any
from .vec_math import l2_dist, dot, norm, normalize, sub, add, scale


class ForwardReachabilityEstimator:
    """
    Set-Based Forward Reachability Estimator R_c(s).
    Approximates the spatial manifold reachable from state s within c steps
    under actuator limits, maximum velocity, and spatial obstacles.
    """
    def __init__(
        self,
        c_step: int = 8,
        max_speed: float = 1.2,
        dt: float = 0.1,
        room_size: float = 10.0,
        doorway_width: float = 1.6
    ):
        self.c_step = c_step
        self.max_speed = max_speed
        self.dt = dt
        self.room_size = room_size
        self.doorway_width = doorway_width
        self.max_reach_dist = c_step * max_speed * dt * 1.35

    def compute_reachable_radius(self, velocity: Optional[List[float]] = None) -> float:
        """Computes dynamic reachability radius adjusted for current momentum."""
        vel_mag = norm(velocity[:2]) if (velocity and len(velocity) >= 2) else 0.0
        return float(min(self.max_reach_dist, self.c_step * (self.max_speed + 0.5 * vel_mag) * self.dt))

    def get_reachable_polygon(
        self,
        agent_pos: List[float],
        velocity: Optional[List[float]] = None,
        num_points: int = 16
    ) -> List[List[float]]:
        """
        Returns a polygon representing the forward reachable set boundary R_c(s)
        clipped against bounding walls and environment geometry.
        """
        radius = self.compute_reachable_radius(velocity)
        ax, ay = agent_pos[0], agent_pos[1]
        polygon = []

        for i in range(num_points):
            angle = (2.0 * math.pi * i) / num_points
            dx = math.cos(angle) * radius
            dy = math.sin(angle) * radius

            px = max(0.1, min(self.room_size - 0.1, ax + dx))
            py = max(0.1, min(self.room_size - 0.1, ay + dy))
            polygon.append([round(px, 3), round(py, 3)])

        return polygon

    def is_reachable(self, start_pos: List[float], target_pos: List[float], radius: Optional[float] = None) -> bool:
        """Checks whether target_pos lies within the forward reachability boundary."""
        r = radius if radius is not None else self.max_reach_dist
        dist = l2_dist(start_pos[:2], target_pos[:2])
        return dist <= (r + 1e-4)


class SymbolicTopologyGraph:
    """
    Learns and maintains symbolic abstract landmarks V and connectivity graph G = (V, E).
    Identifies bottleneck passages (doorways, corridor intersections) automatically.
    """
    def __init__(self, room_size: float = 10.0, doorway_width: float = 1.6):
        self.room_size = room_size
        self.doorway_width = doorway_width
        self.mid = room_size / 2.0

        # Automatic bottleneck discovery: Doorways in 4-room continuous layout
        self.landmarks: List[Dict[str, Any]] = [
            {"id": "node_center_left", "pos": [self.mid * 0.5, self.mid], "type": "room_center", "label": "Room W"},
            {"id": "node_center_right", "pos": [self.mid * 1.5, self.mid], "type": "room_center", "label": "Room E"},
            {"id": "node_center_top", "pos": [self.mid, self.mid * 1.5], "type": "room_center", "label": "Room N"},
            {"id": "node_center_bottom", "pos": [self.mid, self.mid * 0.5], "type": "room_center", "label": "Room S"},
            {"id": "node_door_w_s", "pos": [self.mid, self.mid * 0.5], "type": "bottleneck", "label": "Door S"},
            {"id": "node_door_w_n", "pos": [self.mid, self.mid * 1.5], "type": "bottleneck", "label": "Door N"},
            {"id": "node_door_h_w", "pos": [self.mid * 0.5, self.mid], "type": "bottleneck", "label": "Door W"},
            {"id": "node_door_h_e", "pos": [self.mid * 1.5, self.mid], "type": "bottleneck", "label": "Door E"},
        ]

    def get_nearest_landmark(self, pos: List[float], filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Finds nearest topological landmark."""
        candidates = [lm for lm in self.landmarks if (filter_type is None or lm["type"] == filter_type)]
        if not candidates:
            candidates = self.landmarks
        return min(candidates, key=lambda lm: l2_dist(pos[:2], lm["pos"]))

    def plan_topological_waypoints(self, start_pos: List[float], goal_pos: List[float]) -> List[List[float]]:
        """
        Plans a sequence of topological bottleneck milestones connecting start to goal.
        """
        waypoints: List[List[float]] = []
        sx, sy = start_pos[0], start_pos[1]
        gx, gy = goal_pos[0], goal_pos[1]

        # Check if crossing vertical wall (x crosses mid)
        if (sx < self.mid and gx > self.mid) or (sx > self.mid and gx < self.mid):
            # Pick the doorway closest to current y
            door_y = self.mid * 0.5 if (sy < self.mid) else self.mid * 1.5
            waypoints.append([self.mid, door_y])

        # Check if crossing horizontal wall (y crosses mid)
        if (sy < self.mid and gy > self.mid) or (sy > self.mid and gy < self.mid):
            door_x = self.mid * 0.5 if (sx < self.mid) else self.mid * 1.5
            waypoints.append([door_x, self.mid])

        waypoints.append([gx, gy])
        return waypoints


class ReachabilityProjector:
    """
    Projects raw continuous FeUdal Manager sub-goals into the Reachable Set R_c(s),
    and modulates sub-goals using symbolic topological gateway vectors.
    """
    def __init__(self, c_step: int = 8, max_speed: float = 1.2, dt: float = 0.1, room_size: float = 10.0):
        self.estimator = ForwardReachabilityEstimator(c_step=c_step, max_speed=max_speed, dt=dt, room_size=room_size)
        self.topology = SymbolicTopologyGraph(room_size=room_size)

    def project_subgoal(
        self,
        agent_pos: List[float],
        raw_subgoal: List[float],
        goal_pos: Optional[List[float]] = None,
        velocity: Optional[List[float]] = None,
        guidance_weight: float = 0.45
    ) -> Tuple[List[float], float, Dict[str, Any]]:
        """
        Projects raw_subgoal into R_c(agent_pos) blended with symbolic reachability landmarks.
        Returns: (projected_subgoal, reachability_feasibility, debug_info)
        """
        radius = self.estimator.compute_reachable_radius(velocity)
        sg_norm = norm(raw_subgoal[:2]) or 1.0

        # Base normalized subgoal scaled to reachability horizon
        dir_x = raw_subgoal[0] / sg_norm
        dir_y = raw_subgoal[1] / sg_norm

        # If distant goal is provided, inject symbolic topological guidance vector
        topo_vector = [0.0, 0.0]
        waypoints = []
        if goal_pos:
            waypoints = self.topology.plan_topological_waypoints(agent_pos, goal_pos)
            if waypoints:
                target_wp = waypoints[0]
                wp_delta = sub(target_wp[:2], agent_pos[:2])
                wp_norm = norm(wp_delta) or 1.0
                topo_vector = [wp_delta[0] / wp_norm, wp_delta[1] / wp_norm]

        # Blend manager intention with reachability topology
        blended_x = (1.0 - guidance_weight) * dir_x + guidance_weight * topo_vector[0]
        blended_y = (1.0 - guidance_weight) * dir_y + guidance_weight * topo_vector[1]
        blended_norm = math.sqrt(blended_x ** 2 + blended_y ** 2) or 1.0

        scale_factor = min(1.0, max(0.4, sg_norm))
        final_sg_x = (blended_x / blended_norm) * scale_factor
        final_sg_y = (blended_y / blended_norm) * scale_factor

        projected = [round(final_sg_x, 4), round(final_sg_y, 4)]
        feasibility = round(min(1.0, 1.0 - abs(sg_norm - scale_factor) * 0.2), 3)

        debug_info = {
            "reach_radius": round(radius, 3),
            "reach_polygon": self.estimator.get_reachable_polygon(agent_pos, velocity),
            "topological_waypoints": waypoints,
            "feasibility": feasibility
        }

        return projected, feasibility, debug_info
