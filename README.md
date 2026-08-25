# AGV-Gym

**AGV-Gym: Deep Reinforcement Learning Environment for Battery-Constrained Automated Guided Vehicles Scheduling Problems**

Chey Fung, Li-Pei Wong (Universiti Sains Malaysia)

---

## Overview

**AGV-Gym** is an open-source, Gymnasium-compatible simulation platform designed specifically for evaluating and training Deep Reinforcement Learning (DRL) algorithms on the Automated Guided Vehicle (AGV) Scheduling Problem with Battery Constraints (ASP-BC).

AGV-Gym models the ASP-BC as a **Constrained Markov Decision Process (CMDP)**:
- **Cumulative constraints**: Battery level monitoring and proactive charging at dedicated charging stations bounded by an adaptive non-critical battery threshold ($\bar{b}_v$).
- **Instantaneous constraints**: Critical battery threshold ($\underline{b}_v$) action masking and AGV loading capacity limits ($W_v$).
- **Multi-objective optimization**: Minimization of tardiness penalties (soft time windows) and travel distance costs parameterized by trade-off coefficient $\alpha \in [0, 1]$.
- **Real-world manufacturing benchmark**: Real-world dataset from the Brainport Industries Campus (BIC) in Eindhoven, Netherlands (Singh, 2024).

---

## Default Shop Floor Topology & Dataset (BIC)

- **Total stations**: 78 nodes in a complete directed layout graph $G=(N, A)$.
  - **Pickup & Delivery stations (PD)**: 60 stations
  - **Charging stations (C)**: 9 stations
  - **Parking / Relocation stations (P)**: 9 stations
- **Fleet sizes**: 9, 12, 15, or 18 AGVs
- **Request scenarios**: 
  - ``not-busy``: 450 requests / 24h
  - ``typical``: 900 requests / 24h
  - ``busy``: 1800 requests / 24h
- **Scheduling time horizon**: 24 hours (86,400 seconds)
- **Distance Metric**: Manhattan (cityblock) distance with unit travel speed.

---

## CMDP Formulation

### 1. State Space ($\mathcal{S}$)
The observation dictionary contains:
- ``time``: Current simulated time (seconds $\in [0, 86400]$).
- ``request_locs``: Coordinates $(x, y)$ of origins and destinations of pending requests, shape ``(num_pending, 2, 2)``.
- ``request_times``: Earliest possible pickup / release time $e_r$, shape ``(num_pending,)``.
- ``request_deadlines``: Latest delivery deadline $l_r = e_r + \text{pt}_r + \text{slack}$, shape ``(num_pending,)``.
- ``request_weights``: Loading weight $w_r$, shape ``(num_pending,)``.
- ``v_locs``: Current coordinates $(x, y)$ of all AGVs in the fleet, shape ``(num_vehicles, 2)``.
- ``v_battery``: Current battery percentages $b_v \in [0, 100]\%$, shape ``(num_vehicles,)``.
- ``v_jobs``: Job status queue (``IDLE=0, REPOSITION=1, SETUP=2, PROCESS=3, CHARGING=4, NULL=5``), shape ``(num_vehicles, 3)``.
- ``v_job_locs``: Origin and destination coordinates of queued jobs, shape ``(num_vehicles, 3, 2, 2)``.

### 2. Action Space ($\mathcal{A}$)
The action dictionary contains:
- ``req_assgts``: AGV assigned to each pending request (``0..num_vehicles-1``, or ``num_vehicles`` for unassigned), shape ``(num_pending,)``.
- ``reposition``: Target station for idle AGV repositioning (``0..num_stations-1``, or ``num_stations`` for stay), shape ``(num_vehicles,)``.
- ``req_rejections``: Binary rejection flags, shape ``(num_pending,)``.

### 3. Reward / Objective Function ($\mathcal{R}$)
The CMDP minimizes total operational cost parameterized by $\alpha$:
$$\min \alpha \sum_{v \in V} \sum_{r \in J} c_r \tau_{rv} + (1-\alpha) \sum_{v \in V} \sum_{r \in J} c_v \delta_{rv}$$
Where:
- $\tau_{rv} = \max(0, t^{\text{deliv}}_r - l_r)$ is the tardiness of request $r$.
- $\delta_{rv}$ is the total distance traveled by AGV $v$.
- $c_r$ is the priority / tardiness cost rate per second.
- $c_v$ is the travel distance cost rate.
- $\alpha \in [0, 1]$ controls the trade-off (higher $\alpha$ prioritizes punctuality, lower $\alpha$ prioritizes energy / distance conservation).

Step reward provided to RL agents:
$$R_t = - \left[ \alpha \cdot \text{Tardiness Cost}_t + (1-\alpha) \cdot \text{Travel Cost}_t \right]$$

### 4. Constraints & Battery Management ($\mathcal{C}$)
- **Discharge rate**: $bd_v = 0.0055\%/\text{s}$ during transit and processing.
- **Charging rate**: $bc_v = 0.011\%/\text{s}$ when docked at any of the 9 charging stations.
- **Critical battery threshold**: $\underline{b}_v = 20\%$. AGVs with $b_v < \underline{b}_v$ are instantaneously masked from accepting new transport requests and must recharge.
- **Non-critical threshold**: $\bar{b}_v = 80\%$. Idle AGVs can perform early charging if $b_v \le \bar{b}_v$, and can leave charging stations once $b_v \ge \bar{b}_v$.

---

## Installation & Quick Start

```bash
# Clone the repository
git clone https://github.com/fungchey/AGV-Gym.git
cd AGV-Gym

# Install package
pip install -e .
```

### Running the Heuristic Baseline
```python
from pydispatching import AGVEnv
from algorithm import heuristics

# Initialize environment
env = AGVEnv(num_vehicles=9, num_requests=900, alpha=0.5)
heu = heuristics.Heuristics(env)

obs = env.reset()
terminal = False

while not terminal:
    action = heu.nearest_neighbour(obs)
    obs, reward, terminal, info = env.step(action)

print(f"Total Cost: {env.total_cost:.2f}, Completed Requests: {env.num_completed_requests}")
```

Or run the evaluation script directly:
```bash
# Run headless (3 episodes, default settings)
python simplemain.py

# Run with visual rendering
python simplemain.py --render

# Custom evaluation arguments
python simplemain.py --render --fleet_size 12 --num_requests 450 --alpha 0.75
```

---

## Algorithms Included

| Script | Algorithm Type | Description |
| :--- | :--- | :--- |
| ``simplemain.py`` | **Heuristic (Rule-Based)** | Greedy Nearest Neighbour + Proactive Charging |
| ``algorithm/dqn.py`` | **Deep Q-Network (DQN / Double / Dueling)** | Multi-Head MLP + Action Masking + Experience Replay |

---

## Running Deep Q-Network (DQN / Double DQN / Dueling DQN)

DQN supports high-throughput parallel experience collection across multi-core CPUs (``--num_workers``) with batched GPU Q-inference on CUDA:

```bash
# 1. High-Performance Multi-Worker Vectorized Training (16 Workers, Batch 512, Checkpointing & TensorBoard)
python -m algorithm.dqn --train_steps 200000 --num_workers 16 --batch_size 512 --utd_ratio 1 --alpha 0.9 --save_dir checkpoints/dqn --tensorboard

# 2. Episode-by-Episode Training with Periodic Checkpoint Saving
python -m algorithm.dqn --train_episodes 20 --num_workers 8 --batch_size 256 --save_dir checkpoints/dqn --save_freq 5 --tensorboard

# 3. Resume Training from Checkpoint
python -m algorithm.dqn --resume checkpoints/dqn/dqn_latest.ckpt --train_steps 100000 --num_workers 16 --save_dir checkpoints/dqn

# 4. Evaluate Saved DQN Checkpoint (5 Benchmark Episodes)
python -m algorithm.dqn --load_model checkpoints/dqn/dqn_best.ckpt --eval_episodes 5 --fleet_size 9 --requests 450
```

### DQN CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| ``--train_steps`` | Total environment transitions to train for | — |
| ``--train_episodes`` | Total full episodes to train for (alternative to ``--train_steps``) | — |
| ``--eval_episodes`` | Episodes to run during evaluation | ``1`` |
| ``--num_workers`` | Parallel worker processes for experience collection | ``1`` |
| ``--batch_size`` | Replay buffer mini-batch size | ``64`` |
| ``--utd_ratio`` | Update-to-data ratio (gradient updates per environment step) | ``1`` |
| ``--fleet_size`` | Number of AGVs | ``9`` |
| ``--requests`` | Number of transport requests per episode | ``450`` |
| ``--alpha`` | Tardiness vs. travel cost trade-off ($\alpha \in [0,1]$) | ``0.9`` |
| ``--dueling`` | Enable Dueling DQN architecture | ``False`` |
| ``--double_dqn`` | Enable Double DQN target computation | ``False`` |
| ``--save_dir`` | Directory for checkpoint files | — |
| ``--save_freq`` | Save checkpoint every N episodes | — |
| ``--resume`` | Resume training from ``.ckpt`` file | — |
| ``--load_model`` | Load model weights for evaluation only | — |
| ``--tensorboard`` | Enable TensorBoard logging | ``False`` |
| ``--log_dir`` | TensorBoard log directory | ``runs/`` |
| ``--epsilon_decay`` | Multiplicative epsilon decay per update step | ``0.9999`` |

---

## Running the Heuristic Baseline (``simplemain.py``)

```bash
# 3-episode headless evaluation (default)
python simplemain.py

# With rendering
python simplemain.py --render

# Custom settings
python simplemain.py --num_eval_episodes 5 --fleet_size 12 --num_requests 900 --alpha 0.9
```

### ``simplemain.py`` CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| ``--render`` | Enable visual rendering | ``False`` |
| ``--num_eval_episodes`` | Number of evaluation episodes | ``3`` |
| ``--fleet_size`` | Number of AGVs | ``9`` |
| ``--num_requests`` | Number of transport requests per episode | ``900`` |
| ``--alpha`` | Tardiness vs. travel cost trade-off ($\alpha \in [0,1]$) | ``0.9`` |

---

## Model Checkpointing & Reproducibility (``.ckpt``)

DQN provides full PyTorch checkpoint saving (``.ckpt``) containing complete model parameters, optimizer momentum states, and environment configurations for scientific reproducibility:

#### Checkpoint File Naming Conventions
When ``--save_dir <dir>`` is specified:
- ``dqn_best.ckpt``: Saved whenever evaluation achieves a new best completion rate or lowest operational cost.
- ``dqn_latest.ckpt``: Updated at the end of each training run.
- ``dqn_ep_<N>.ckpt``: Periodic snapshot saved when ``--save_freq <N>`` is set.

#### Checkpoint Structure
```python
{
    "format_version": "1.0",
    "timestamp": 1740478000.0,
    "algorithm": "DQN",
    "model_state_dict": {...},           # Neural network weights (PyTorch state_dict)
    "optimizer_state_dict": {...},       # Adam optimizer state
    "total_episodes": 20,               # Total completed episodes
    "total_env_steps": 200000,          # Total environment transitions
    "total_train_updates": 400,         # Total gradient update steps
    "model_config": {                   # Architecture configuration
        "hidden_dim": 256,
        "dueling": True,
        "double_dqn": True
    },
    "hyperparameters": {                # Training hyperparameters
        "lr": 1e-4, "gamma": 0.99, "alpha": 0.9
    },
    "best_eval_cost": 12450.0,          # Best evaluation cost achieved
    "best_eval_completion_pct": 98.4,   # Best delivery completion percentage
    "eval_stats": {...}                 # Detailed metric breakdown
}
```

---

## Training Monitoring with TensorBoard

```bash
tensorboard --logdir runs
```
Open ``http://localhost:6006`` in your browser to inspect live metrics:

* **Episodic Rollout Metrics** (X-axis = ``Episode Number``):
  * ``Rollout/completion_rate_pct``: Percentage of requests successfully delivered.
  * ``Rollout/completed_requests``: Count of fulfilled transport orders.
  * ``Rollout/stranded_agvs``: Count of vehicles immobilized due to battery depletion ($b_v \le 0\%$).
  * ``Rollout/episode_cost``: Total operational cost ($C = \alpha \cdot \text{Tardiness} + (1-\alpha) \cdot \text{Travel}$).
  * ``Rollout/travel_distance``: Total physical distance traversed by the fleet.
  * ``Rollout/tardiness_cost``: Total lateness penalty incurred.
  * ``Rollout/episode_reward``: Undiscounted cumulative episode reward.

* **Step-Level Metrics** (X-axis = ``Environment Step``):
  * ``Rollout/step_reward``: Instantaneous step reward.
  * ``Epsilon/exploration_rate``: DQN epsilon exploration probability.
  * ``Loss/total_loss``: Combined TD loss.
  * ``Loss/q_joint_mean``: Mean Q-value across the joint action space.
  * ``Loss/td_error``: Temporal difference error.

* **Evaluation Summaries** (Logged per evaluation run):
  * ``Eval/mean_cost``, ``Eval/completion_rate_pct``, ``Eval/mean_stranded_agvs``, ``Eval/mean_tardiness``, ``Eval/mean_dist``, ``Eval/mean_reward``.

---

## Running Unit Test Suites

```bash
python -m unittest test_env.py     # Environment core dynamics & constraints
python -m unittest test_dqn.py     # Deep Q-Network (DQN) test suite
```

---

## Citation
If you use AGV-Gym in your research, please cite:
```bibtex
@article{fung2026agvgym,
  title={AGV-Gym: Deep reinforcement learning environment for battery-constrained automated guided vehicles scheduling problems},
  author={Fung, Chey and Wong, Li-Pei},
  journal={Preprint submitted to Elsevier},
  year={2026}
}
```
