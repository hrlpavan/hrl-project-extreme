# HRL Project Extreme

> High-Capacity Hierarchical Reinforcement Learning Engine for Long-Horizon Autonomy  
> Engineered with FeUdal Networks (FuN), Dual-Timescale Macro/Micro Policies, Continuous Control (MuJoCo/Gymnasium support), Parallel Vectorized Rollout Workers, and an Interactive Web Visualizer Dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![HRL: FeUdal](https://img.shields.io/badge/RL-Hierarchical-purple.svg)]()
[![Continuous Control](https://img.shields.io/badge/Control-Continuous-green.svg)]()
[![Web Visualizer](https://img.shields.io/badge/UI-Interactive_Dashboard-cyan.svg)]()

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Environment State s_t     |
                      +--------------+--------------+
                                     |
               (Every C steps)       |       (Every step)
                      v              |             v
          +-----------------------+  |  +-----------------------+
          |     Manager Policy    |  |  |     Worker Policy     |
          |  (Directional Goal g) |  |  |   (Continuous Force / |
          |   V^M(s_t) Baseline   |  |  |    Discrete Action)   |
          +-----------+-----------+  |  +-----------+-----------+
                      |              |              |
                      +------------> | <------------+
                                     v
                        +---------------------------+
                        | Intrinsic Motivation      |
                        | r_i = cos(Δs, g) - 0.5|d| |
                        +---------------------------+
                                     |
                        +---------------------------+
                        | Parallel Vectorized Envs  |
                        |   (Workers 1 ... N)       |
                        +---------------------------+
                                     |
                        +---------------------------+
                        | Interactive Web Dashboard |
                        |  (Real-Time Visualizer)   |
                        +---------------------------+
```

---

## Quickstart

### 1. Launch Interactive Web Visualizer Dashboard
```bash
python3 -m hrl_extreme.server --port 8000
# Or via CLI dispatcher:
python3 -m hrl_extreme web --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to interact with the live simulation arena, inspect directional sub-goal projections $g_t$, watch multi-worker vector rollouts, and view real-time Chart.js telemetry.

### 2. Run Continuous Control Training with 4 Parallel Workers
```bash
python3 -m hrl_extreme.train --mode continuous --vec-envs 4 --episodes 100
```

### 3. Run Discrete Maze Benchmark Training
```bash
python3 -m hrl_extreme.train --mode discrete --vec-envs 4 --episodes 100
```

### 4. Run Trajectory Evaluation and Rollouts
```bash
python3 -m hrl_extreme.evaluate --mode continuous --episodes 5
```

### 5. Run Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## Project Layout

```
hrl-project-extreme/
├── hrl_extreme/
│   ├── __init__.py           # Package entry point & version exports
│   ├── __main__.py           # CLI command dispatcher (web, train, evaluate)
│   ├── agent.py              # Dual-mode Hierarchical FeUdal Agent
│   ├── buffer.py             # Dual-Timescale Replay Buffer with GAE
│   ├── env.py                # Discrete Sparse-Reward Bottleneck Maze
│   ├── continuous_env.py     # Continuous 2D Physics Navigation Benchmark
│   ├── gym_wrapper.py        # Gymnasium & MuJoCo Environment Adapter
│   ├── vec_env.py            # Parallel Vectorized Environment Manager
│   ├── models.py             # Native Continuous & Discrete Actor-Critic Models
│   ├── torch_models.py       # PyTorch Deep Policy Optimization Networks
│   ├── vec_math.py           # Vectorized Math, Statistics & GAE Engine
│   ├── train.py              # Vectorized Training Pipeline & Metric Logger
│   ├── evaluate.py           # Trajectory Evaluation & Benchmark Visualizer
│   ├── server.py             # Starlette / Uvicorn Live Visualizer Web Server
│   └── static/
│       └── index.html        # Modern Dark Canvas & Chart.js Web Dashboard
├── tests/
│   └── test_hrl.py           # Comprehensive Unit & Integration Test Suite
├── PROJECT_BLUEPRINT.md      # Long-term vision and phase roadmaps
└── README.md                 # Project documentation
```

---

## License
MIT License.
