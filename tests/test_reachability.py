import unittest
from hrl_extreme.reachability import ForwardReachabilityEstimator, SymbolicTopologyGraph, ReachabilityProjector
from hrl_extreme.agent import HierarchicalAgent
from hrl_extreme.continuous_env import ContinuousGoalNavigationEnv
from hrl_extreme.env import SparseGoalMazeEnv


class TestReachabilityAnalysis(unittest.TestCase):
    def setUp(self):
        self.estimator = ForwardReachabilityEstimator(c_step=8, max_speed=1.2, dt=0.1, room_size=10.0)
        self.topology = SymbolicTopologyGraph(room_size=10.0)
        self.projector = ReachabilityProjector(c_step=8, room_size=10.0)

    def test_reachability_radius_and_polygon(self):
        r = self.estimator.compute_reachable_radius(velocity=[0.5, 0.5])
        self.assertGreater(r, 0.5)
        self.assertLessEqual(r, self.estimator.max_reach_dist)

        poly = self.estimator.get_reachable_polygon([5.0, 5.0], [0.0, 0.0], num_points=8)
        self.assertEqual(len(poly), 8)
        for pt in poly:
            self.assertGreaterEqual(pt[0], 0.0)
            self.assertLessEqual(pt[0], 10.0)
            self.assertGreaterEqual(pt[1], 0.0)
            self.assertLessEqual(pt[1], 10.0)

    def test_symbolic_topology_waypoints(self):
        # Start in bottom-left [1.5, 1.5] and goal in top-right [8.5, 8.5]
        waypoints = self.topology.plan_topological_waypoints([1.5, 1.5], [8.5, 8.5])
        self.assertGreater(len(waypoints), 0)
        # Should include doorways crossing mid line (5.0)
        has_mid_x = any(abs(wp[0] - 5.0) < 1e-3 for wp in waypoints)
        has_mid_y = any(abs(wp[1] - 5.0) < 1e-3 for wp in waypoints)
        self.assertTrue(has_mid_x or has_mid_y)

    def test_reachability_projection(self):
        raw_sg = [2.0, 2.0]
        proj_sg, feas, debug = self.projector.project_subgoal(
            agent_pos=[1.5, 1.5],
            raw_subgoal=raw_sg,
            goal_pos=[8.5, 8.5],
            velocity=[0.2, 0.2]
        )
        self.assertEqual(len(proj_sg), 2)
        self.assertGreater(feas, 0.0)
        self.assertIn("reach_polygon", debug)
        self.assertIn("topological_waypoints", debug)

    def test_agent_with_reachability(self):
        agent = HierarchicalAgent(obs_dim=6, subgoal_dim=2, action_dim=2, continuous=True, use_reachability=True)
        env = ContinuousGoalNavigationEnv()
        obs = env.reset()

        action, sg, lp, val = agent.select_action(obs, goal_pos=env.goal)
        self.assertEqual(len(action), 2)
        self.assertEqual(len(sg), 2)
        self.assertGreater(agent.reachability_feasibility, 0.0)
        self.assertTrue(agent.use_reachability)


if __name__ == "__main__":
    unittest.main()
