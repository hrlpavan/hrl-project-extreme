import os
import time
import json
import argparse
from typing import Dict, Any, List

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .continuous_env import ContinuousGoalNavigationEnv
from .env import SparseGoalMazeEnv
from .vec_env import make_vec_env
from .agent import HierarchicalAgent

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
ASSETS_DIR = os.path.join(STATIC_DIR, "assets")

class RLServerState:
    def __init__(self):
        self.mode = "continuous"
        self.c_step = 8
        self.num_vec_workers = 4
        self.total_episodes = 0
        self.total_steps = 0
        self.recent_successes = []
        self.recent_rewards = []
        self.t_last_step = time.time()
        self.step_fps = 30.0

        self.agent = None
        self.vec_env = None
        self.obs_list = None
        self.steps_in_subgoal = [0] * self.num_vec_workers
        self.current_subgoals = [[0.0, 0.0] for _ in range(self.num_vec_workers)]
        self.manager_start_obs = None
        self.cumulative_rewards = [0.0] * self.num_vec_workers
        self.trajectories = [[] for _ in range(self.num_vec_workers)]

        self.reset(mode="continuous", c_step=8)

    def reset(self, mode: str = "continuous", c_step: int = 8):
        self.mode = mode
        self.c_step = c_step
        is_continuous = (mode == "continuous")
        env_class = ContinuousGoalNavigationEnv if is_continuous else SparseGoalMazeEnv

        self.vec_env = make_vec_env(env_class, num_envs=self.num_vec_workers)
        self.agent = HierarchicalAgent(
            obs_dim=self.vec_env.observation_dim,
            subgoal_dim=self.vec_env.subgoal_dim,
            action_dim=self.vec_env.action_dim,
            continuous=is_continuous,
            c_step=c_step,
            use_torch=True,
        )

        self.obs_list = self.vec_env.reset()
        self.steps_in_subgoal = [0] * self.num_vec_workers
        self.current_subgoals = [[0.0] * self.vec_env.subgoal_dim for _ in range(self.num_vec_workers)]
        self.manager_start_obs = [list(obs) for obs in self.obs_list]
        self.cumulative_rewards = [0.0] * self.num_vec_workers

        # Extract initial positions
        self.trajectories = []
        for env in self.vec_env.envs:
            pos = getattr(env, "pos", getattr(env, "agent_pos", [1.5, 1.5]))
            self.trajectories.append([[round(float(p), 2) for p in pos]])

        return self.get_state_dict()

    def step(self, train: bool = True) -> Dict[str, Any]:
        t_now = time.time()
        dt = t_now - self.t_last_step
        if dt > 0:
            self.step_fps = 0.9 * self.step_fps + 0.1 * (1.0 / dt)
        self.t_last_step = t_now

        # 1. Action selection
        actions, self.current_subgoals, log_probs, values = self.agent.select_actions_vec(
            obs_list=self.obs_list,
            steps_in_subgoal=self.steps_in_subgoal,
            current_subgoals=self.current_subgoals,
            evaluate=False,
        )

        # 2. Vector step
        next_obs_list, rewards, dones, infos = self.vec_env.step(actions)
        self.total_steps += 1

        primary_r_i = 0.0
        losses = None

        for i in range(self.num_vec_workers):
            self.steps_in_subgoal[i] += 1
            self.cumulative_rewards[i] += rewards[i]
            r_i = self.agent.compute_intrinsic_reward(self.obs_list[i], next_obs_list[i], self.current_subgoals[i])
            if i == 0:
                primary_r_i = r_i

            # Update trajectory
            pos = infos[i].get("agent_pos", self.obs_list[i][:2])
            self.trajectories[i].append([round(float(p), 2) for p in pos])
            if len(self.trajectories[i]) > 80:
                self.trajectories[i].pop(0)

            # Store transitions
            self.agent.buffer.store_worker(
                obs=self.obs_list[i],
                subgoal=self.current_subgoals[i],
                action=actions[i],
                intrinsic_reward=r_i,
                next_obs=next_obs_list[i],
                done=dones[i],
                log_prob=log_probs[i],
                value=values[i],
            )

            if self.steps_in_subgoal[i] >= self.c_step or dones[i]:
                self.agent.buffer.store_manager(
                    obs=self.manager_start_obs[i],
                    subgoal=self.current_subgoals[i],
                    cumulative_extrinsic_reward=self.cumulative_rewards[i],
                    next_obs=next_obs_list[i],
                    done=dones[i],
                )
                self.manager_start_obs[i] = list(next_obs_list[i])
                self.cumulative_rewards[i] = 0.0
                self.steps_in_subgoal[i] = 0

            if dones[i]:
                self.total_episodes += 1
                is_succ = infos[i].get("is_success", False)
                self.recent_successes.append(1 if is_succ else 0)
                self.recent_rewards.append(self.cumulative_rewards[i])
                if len(self.recent_successes) > 30:
                    self.recent_successes.pop(0)
                if len(self.recent_rewards) > 30:
                    self.recent_rewards.pop(0)
                self.trajectories[i] = [[round(float(p), 2) for p in infos[i].get("agent_pos", [1.5, 1.5])]]

        self.obs_list = next_obs_list

        # Online training update
        if train and len(self.agent.buffer) >= 16:
            losses = self.agent.train_batch(batch_size=16)

        state = self.get_state_dict()
        state.update({
            "action": actions[0] if isinstance(actions[0], list) else [float(actions[0]), 0.0],
            "reward": float(rewards[0]),
            "intrinsic_reward": float(primary_r_i),
            "step_in_subgoal": int(self.steps_in_subgoal[0]),
            "dist_to_goal": float(infos[0].get("dist_to_goal", 0.0)),
            "velocity": [round(float(v), 2) for v in infos[0].get("velocity", [0.0, 0.0])],
            "is_success": bool(infos[0].get("is_success", False)),
            "done": bool(dones[0]),
            "losses": losses,
        })
        return state

    def get_state_dict(self) -> Dict[str, Any]:
        p_env = self.vec_env.envs[0]
        room_size = getattr(p_env, "room_size", getattr(p_env, "grid_size", 10.0))
        agent_pos = getattr(p_env, "pos", getattr(p_env, "agent_pos", [1.5, 1.5]))
        goal_pos = getattr(p_env, "goal", getattr(p_env, "goal_pos", [room_size - 1.5, room_size - 1.5]))

        sr = (sum(self.recent_successes) / max(1, len(self.recent_successes))) * 100.0 if self.recent_successes else 0.0

        # Vector workers state
        vector_states = []
        for i, env in enumerate(self.vec_env.envs):
            v_pos = getattr(env, "pos", getattr(env, "agent_pos", [1.5, 1.5]))
            v_goal = getattr(env, "goal", getattr(env, "goal_pos", [room_size - 1.5, room_size - 1.5]))
            vector_states.append({
                "worker_id": i + 1,
                "agent_pos": [round(float(p), 2) for p in v_pos],
                "goal_pos": [round(float(p), 2) for p in v_goal],
                "subgoal": [round(float(g), 3) for g in self.current_subgoals[i]],
                "trajectory": self.trajectories[i],
            })

        # Reachability Analysis Telemetry (GARA / STAR)
        reach_poly = []
        feasibility = 1.0
        topo_landmarks = []
        topo_waypoints = []
        use_reach = True

        if self.agent and hasattr(self.agent, "reachability_projector"):
            use_reach = getattr(self.agent, "use_reachability", True)
            feasibility = getattr(self.agent, "reachability_feasibility", 1.0)
            reach_poly = self.agent.reachability_projector.estimator.get_reachable_polygon(
                agent_pos=agent_pos,
                velocity=getattr(p_env, "velocity", [0.0, 0.0])
            )
            topo_landmarks = self.agent.reachability_projector.topology.landmarks
            topo_waypoints = self.agent.reachability_debug.get("topological_waypoints", [])

        return {
            "mode": self.mode,
            "room_size": float(room_size),
            "agent_pos": [round(float(p), 2) for p in agent_pos],
            "goal_pos": [round(float(p), 2) for p in goal_pos],
            "subgoal": [round(float(g), 3) for g in self.current_subgoals[0]],
            "fps": float(self.step_fps),
            "success_rate": float(sr),
            "buffer_size": len(self.agent.buffer) if self.agent else 0,
            "total_episodes": self.total_episodes,
            "vector_states": vector_states,
            "reachability_polygon": reach_poly,
            "reachability_feasibility": round(float(feasibility), 3),
            "topological_landmarks": topo_landmarks,
            "topological_waypoints": topo_waypoints,
            "use_reachability": use_reach,
        }

server_state = RLServerState()

async def handle_index(request):
    index_file = os.path.join(STATIC_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)

async def handle_reset(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    mode = data.get("mode", "continuous")
    c_step = int(data.get("c_step", 8))
    result = server_state.reset(mode=mode, c_step=c_step)
    return JSONResponse(result)

async def handle_step(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    train = data.get("train", True)
    result = server_state.step(train=train)
    return JSONResponse(result)

async def handle_train_batch(request):
    losses = server_state.agent.train_batch(batch_size=32)
    return JSONResponse({"losses": losses, "buffer_size": len(server_state.agent.buffer)})

async def handle_toggle_reachability(request):
    try:
        data = await request.json()
        enabled = data.get("enabled", True)
    except Exception:
        enabled = True
    if server_state.agent:
        server_state.agent.use_reachability = enabled
    return JSONResponse({"use_reachability": enabled})

async def handle_status(request):
    return JSONResponse(server_state.get_state_dict())

routes = [
    Route("/", handle_index, methods=["GET"]),
    Route("/api/reset", handle_reset, methods=["POST"]),
    Route("/api/step", handle_step, methods=["POST"]),
    Route("/api/train_batch", handle_train_batch, methods=["POST"]),
    Route("/api/toggle_reachability", handle_toggle_reachability, methods=["POST"]),
    Route("/api/status", handle_status, methods=["GET"]),
]

if os.path.isdir(ASSETS_DIR):
    routes.append(Mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets"))

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(routes=routes, middleware=middleware)

def run_server(host: str = "127.0.0.1", port: int = 8000):
    print("=" * 75)
    print("   HRL INTERNATIONAL PRIVATE LIMITED™ — HRL PROJECT EXTREME DASHBOARD   ")
    print("=" * 75)
    print(f"  Live Visualizer URL : http://{host}:{port}")
    print("  Brand Standard      : HRL International Master Brand Design System")
    print("  Engine Architecture : FeUdal Networks (FuN) Dual-Timescale Policy Engine")
    print("  Motto               : 'We Can Do Everything Related To Software Sector Without Any Excuses!'")
    print("=" * 75 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRL Extreme Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
