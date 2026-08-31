# HRL Project Extreme 🧠⚡

> **Hierarchical Reinforcement Learning Engine for Long-Horizon Sparse-Reward Autonomy**  
> *Engineered with FeUdal Networks (FuN), Dual-Timescale Macro/Micro Policies, and Goal-Conditioned Intrinsic Motivation.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![HRL: FeUdal](https://img.shields.io/badge/RL-Hierarchical-purple.svg)]()

---

## 🎯 Architecture Overview

```
                      +-----------------------------+
                      |   Environment State s_t     |
                      +--------------+--------------+
                                     |
               (Every C steps)       |       (Every step)
                      v              |             v
          +-----------------------+  |  +-----------------------+
          |     Manager Policy    |  |  |     Worker Policy     |
          |  (Directional Goal g) |  |  |   (Primitive Action)  |
          +-----------+-----------+  |  +-----------+-----------+
                      |              |              |
                      +------------> | <------------+
                                     v
                       +---------------------------+
                       | Intrinsic Reward Engine   |
                       |  r_i = cos_sim(Δs, g)     |
                       +---------------------------+
```

### Key Highlights:
1. **Dual-Timescale Control:**
   * **High-Level Manager:** Operates at interval $C = 8$, learning long-horizon state transitions and setting directional sub-goals.
   * **Low-Level Worker:** Operates at single-step frequency, executing fast motor/discrete primitives conditioned on the sub-goal.
2. **Intrinsic Motivation:** Worker is rewarded via cosine alignment and distance minimization in state-space transitions without needing external sparse rewards.
3. **Hierarchical Experience Replay Buffer:** Dual memory queues separating fast worker experiences from macro manager trajectories.

---

## 🚀 Quickstart

### 1. Run HRL Training Loop
```bash
python3 -m hrl_extreme.train
```

### 2. Run Trajectory Evaluation & Rollouts
```bash
python3 -m hrl_extreme.evaluate
```

---

## 📂 Project Layout

```
hrl-project-extreme/
├── hrl_extreme/
│   ├── __init__.py       # Package entry point
│   ├── agent.py          # Hierarchical FeUdal Agent
│   ├── buffer.py         # Dual-Timescale Replay Buffer
│   ├── env.py            # Sparse-Reward Bottleneck Maze Environment
│   ├── models.py         # Manager & Worker Actor-Critic Networks
│   ├── train.py          # Training loop and metric logger
│   └── evaluate.py       # Trajectory evaluator
└── README.md             # Project documentation
```

---

## 📄 License
MIT License.
