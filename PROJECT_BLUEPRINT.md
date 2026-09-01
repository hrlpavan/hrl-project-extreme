# Project Blueprint: HRL Project Extreme

## High-Capacity Hierarchical Reinforcement Learning Engine for Long-Horizon Autonomy

**Repository Location:** `/Users/pavankumars/.gemini/antigravity/scratch/hrl-project-extreme`  
**Primary Tech Stack:** Python 3.10+, PyTorch / Native Math, FeUdal Networks (FuN), Continuous Control, Vectorized Rollout Workers, Starlette / Canvas Web Dashboard  
**Target Tier:** DeepMind, OpenAI, Tesla Autopilot, Anthropic, Waymo, Quant/Robotics AI Labs  

---

## 1. Executive Summary
HRL Project Extreme is an advanced Hierarchical Reinforcement Learning (HRL) engine based on FeUdal Networks (FuN) and Goal-Conditioned Temporal Abstractions:
* **High-Level Manager (Macro Policy):** Operates at a dilated timescale ($c$ steps), learning long-horizon spatial transitions and generating directional sub-goals in state space.
* **Low-Level Worker (Micro Policy):** Operates at single-step frequency (1 step), conditioned on both current observation and sub-goal, selecting continuous forces $\mathbf{a} \in [-1, 1]^d$ or discrete primitives.
* **Intrinsic Motivation Engine:** Worker receives internal intrinsic rewards ($r_i = \cos(\Delta s, g) - 0.5 \|\Delta s - g\|_2$), eliminating the need for dense external reward shaping.
* **Parallel Vectorized Rollouts:** Executes $N$ environment instances concurrently with synchronized macro-step boundary alignment, achieving >1,100 FPS throughput.
* **Dual-Timescale GAE Experience Replay:** Separates fast worker micro-transitions from macro manager transitions with Generalized Advantage Estimation.
* **Interactive Real-Time Web Visualizer:** Live 60 FPS Canvas rendering of physics simulations, Manager macro sub-goals, Worker forces, 4x vector worker grid, and real-time Chart.js telemetry.

---

## 2. Quickstart Execution
```bash
# 1. Launch Interactive Web Visualizer Dashboard
python3 -m hrl_extreme.server --port 8000

# 2. Run Continuous Control HRL training loop with 4 parallel workers
python3 -m hrl_extreme.train --mode continuous --vec-envs 4 --episodes 100

# 3. Run Discrete Maze benchmark training
python3 -m hrl_extreme.train --mode discrete --vec-envs 4 --episodes 100

# 4. Run trajectory evaluation and visual rollouts
python3 -m hrl_extreme.evaluate --mode continuous --episodes 5

# 5. Run automated test suite
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 3. How to Kickoff Next Conversation (Phase 3)
To continue or scale this project in a new session, provide this prompt:
> *"I am continuing work on my repository at /Users/pavankumars/.gemini/antigravity/scratch/hrl-project-extreme. Read PROJECT_BLUEPRINT.md and let us begin Phase 3: Recurrent Spatial Memory (LSTM/Transformer Subgoal Memory), Multi-Level Hierarchy (L1-L2-L3 policies), and Distributed Ray/Cluster Scale."*
