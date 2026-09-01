import unittest
import math
from hrl_extreme.vec_math import (
    dot, norm, normalize, l2_dist, sub, add, scale, clip,
    cosine_similarity, softmax, gaussian_sample, gaussian_log_prob, compute_gae
)
from hrl_extreme.continuous_env import ContinuousGoalNavigationEnv
from hrl_extreme.env import SparseGoalMazeEnv
from hrl_extreme.gym_wrapper import GymEnvWrapper
from hrl_extreme.vec_env import VectorEnv, make_vec_env
from hrl_extreme.buffer import HierarchicalReplayBuffer
from hrl_extreme.models import ManagerNetwork as NativeManager, WorkerNetwork as NativeWorker
from hrl_extreme.agent import HierarchicalAgent

class TestHRLProjectExtreme(unittest.TestCase):

    def test_vector_math_and_gae(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        v3 = [1.0, 0.0]

        self.assertAlmostEqual(dot(v1, v2), 0.0)
        self.assertAlmostEqual(norm(v1), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)
        self.assertEqual(l2_dist([0, 0], [3, 4]), 5.0)

        # Gaussian sampling and log prob
        mean = [0.0, 1.0]
        std = [0.5, 0.5]
        s = gaussian_sample(mean, std)
        self.assertEqual(len(s), 2)
        lp = gaussian_log_prob(mean, mean, std)
        self.assertTrue(math.isfinite(lp))

        # GAE
        rewards = [1.0, 1.0, 10.0]
        values = [0.5, 1.0, 5.0, 0.0]
        dones = [False, False, True]
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        self.assertEqual(len(adv), 3)
        self.assertEqual(len(ret), 3)

    def test_continuous_env(self):
        env = ContinuousGoalNavigationEnv(room_size=10.0, max_steps=50)
        obs = env.reset(seed=123)
        self.assertEqual(len(obs), 6)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in obs))

        # Test step
        action = [0.5, 0.5]
        next_obs, rew, done, info = env.step(action)
        self.assertEqual(len(next_obs), 6)
        self.assertIn("is_success", info)
        self.assertIn("dist_to_goal", info)
        self.assertIn("agent_pos", info)

    def test_gym_wrapper(self):
        wrapper = GymEnvWrapper(env_id="PointMaze_UMaze-v3", max_steps=20)
        obs = wrapper.reset()
        self.assertEqual(len(obs), wrapper.observation_dim)

        action = [0.0] * wrapper.action_dim
        next_obs, rew, done, info = wrapper.step(action)
        self.assertEqual(len(next_obs), wrapper.observation_dim)

    def test_vector_env(self):
        num_workers = 4
        vec_env = make_vec_env(ContinuousGoalNavigationEnv, num_envs=num_workers, max_steps=10)
        obs_list = vec_env.reset()
        self.assertEqual(len(obs_list), num_workers)
        self.assertEqual(len(obs_list[0]), 6)

        actions = [[0.2, -0.2] for _ in range(num_workers)]
        next_obs_list, rewards, dones, infos = vec_env.step(actions)
        self.assertEqual(len(next_obs_list), num_workers)
        self.assertEqual(len(rewards), num_workers)
        self.assertEqual(len(dones), num_workers)
        self.assertEqual(len(infos), num_workers)
        vec_env.close()

    def test_buffer_operations(self):
        buf = HierarchicalReplayBuffer(capacity=100)
        obs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sg = [0.1, 0.1]
        buf.store_worker(obs, sg, [0.5, -0.5], 1.0, obs, False, 0.0, 1.0)
        self.assertEqual(len(buf), 1)

        buf.store_manager(obs, sg, 5.0, obs, False, 0.0, 2.0)
        self.assertEqual(len(buf.manager_buffer), 1)

        # Batch sampling
        for _ in range(35):
            buf.store_worker(obs, sg, [0.1, 0.2], 0.5, obs, False)
            buf.store_manager(obs, sg, 2.0, obs, False)

        w_batch = buf.sample_worker(batch_size=16)
        m_batch = buf.sample_manager(batch_size=16)
        self.assertIsNotNone(w_batch)
        self.assertEqual(len(w_batch), 16)
        self.assertIsNotNone(m_batch)
        self.assertEqual(len(m_batch), 16)

    def test_native_actor_critics(self):
        mgr = NativeManager(obs_dim=6, subgoal_dim=2)
        obs = [0.1] * 6
        sg, lp, val = mgr.sample_subgoal(obs)
        self.assertEqual(len(sg), 2)
        self.assertTrue(math.isfinite(val))

        wrk_cont = NativeWorker(obs_dim=6, subgoal_dim=2, action_dim=2, continuous=True)
        act, lp, val = wrk_cont.sample_action(obs, sg)
        self.assertEqual(len(act), 2)
        self.assertTrue(all(-1.0 <= a <= 1.0 for a in act))

        # Training step
        loss = wrk_cont.train_step(obs, sg, act, advantage=1.0, target_value=1.5)
        self.assertTrue(math.isfinite(loss))

    def test_agent_training_loop_continuous(self):
        agent = HierarchicalAgent(
            obs_dim=6,
            subgoal_dim=2,
            action_dim=2,
            continuous=True,
            c_step=4,
            use_torch=False,
        )
        env = ContinuousGoalNavigationEnv(max_steps=20)
        obs = env.reset()

        for _ in range(15):
            action, subgoal, lp, val = agent.select_action(obs)
            next_obs, rew, done, info = env.step(action)
            agent.step_update(obs, action, rew, next_obs, done, lp, val)
            obs = next_obs
            if done:
                obs = env.reset()

        losses = agent.train_batch(batch_size=4)
        self.assertIn("worker_loss", losses)
        self.assertIn("manager_loss", losses)
        self.assertTrue(math.isfinite(losses["worker_loss"]))

    def test_agent_training_loop_discrete(self):
        agent = HierarchicalAgent(
            obs_dim=4,
            subgoal_dim=2,
            action_dim=4,
            continuous=False,
            c_step=4,
            use_torch=False,
        )
        env = SparseGoalMazeEnv(max_steps=20)
        obs = env.reset()

        for _ in range(15):
            action, subgoal, lp, val = agent.select_action(obs)
            next_obs, rew, done, info = env.step(action)
            agent.step_update(obs, action, rew, next_obs, done, lp, val)
            obs = next_obs
            if done:
                obs = env.reset()

        losses = agent.train_batch(batch_size=4)
        self.assertIn("worker_loss", losses)
        self.assertIn("manager_loss", losses)

    def test_web_server_endpoints(self):
        from starlette.testclient import TestClient
        from hrl_extreme.server import app

        client = TestClient(app)
        res_index = client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn("HRL Project Extreme", res_index.text)

        res_reset = client.post("/api/reset", json={"mode": "continuous", "c_step": 8})
        self.assertEqual(res_reset.status_code, 200)
        data = res_reset.json()
        self.assertEqual(data["mode"], "continuous")
        self.assertIn("agent_pos", data)

        res_step = client.post("/api/step", json={"train": True})
        self.assertEqual(res_step.status_code, 200)
        step_data = res_step.json()
        self.assertIn("action", step_data)
        self.assertIn("intrinsic_reward", step_data)

        res_train = client.post("/api/train_batch")
        self.assertEqual(res_train.status_code, 200)

        res_status = client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)

if __name__ == "__main__":
    unittest.main()
