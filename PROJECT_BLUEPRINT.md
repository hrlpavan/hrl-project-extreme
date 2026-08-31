# Project Blueprint: HRL Project Extreme

## High-Capacity Hierarchical Reinforcement Learning Engine for Long-Horizon Autonomy

**Repository Location:** `/Users/pavankumars/.gemini/antigravity/scratch/hrl-project-extreme`  
**Primary Tech Stack:** Python 3.10+, PyTorch / Native Math, FeUdal Networks (FuN), Option-Critic Architecture, Dual-Timescale Replay Buffers  
**Target Tier:** DeepMind, OpenAI, Tesla Autopilot, Anthropic, Waymo, Quant/Robotics AI Labs  

---

## 1. Executive Summary
HRL Project Extreme is an advanced Hierarchical Reinforcement Learning (HRL) engine based on FeUdal Networks (FuN) and Goal-Conditioned Temporal Abstractions:
* **High-Level Manager (Macro Policy):** Operates at a dilated timescale (c steps), learning long-horizon spatial transitions and generating directional sub-goals.
* **Low-Level Worker (Micro Policy):** Operates at single-step frequency (1 step), conditioned on both current observation and sub-goal, selecting primitive actions.
* **Intrinsic Motivation Engine:** Worker receives internal intrinsic rewards, eliminating the need for dense external reward shaping.
* **Dual-Timescale Experience Replay:** Separates fast worker transitions from macro manager transitions.

---

## 2. Quickstart Execution
```bash
# 1. Run HRL training loop
PYTHONPATH=. python3 -m hrl_extreme.train

# 2. Run trajectory evaluation and rollouts
PYTHONPATH=. python3 -m hrl_extreme.evaluate
```

---

## 3. How to Kickoff Next Conversation
To continue or scale this project in a new session, provide this initial prompt:
> *"I have initialized the hrl-project-extreme repository at /Users/pavankumars/.gemini/antigravity/scratch/hrl-project-extreme. Read PROJECT_BLUEPRINT.md and let us begin Phase 2: PyTorch Deep Policy Optimization, Continuous Control (MuJoCo / Gymnasium), and Vectorized Rollout Workers."*
