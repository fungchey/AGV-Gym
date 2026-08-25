import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
import argparse
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

from pydispatching.agv_env import AGVEnv
from pydispatching.core import Jobs
from algorithm.vec_env import SubprocVecAGVEnv, EnvFactory

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class DQNObsEncoder:
    """
    Encodes AGV-Gym observation dictionary into a fixed-size flat vector for Deep Q-Network (DQN).
    Supports variable fleet sizes (V) and dynamic pending requests with fixed slot padding (max_slots).
    """

    def __init__(self, env: AGVEnv, max_req_slots: int = 15):
        self.num_vehicles = env.num_vehicles
        self.num_stations = env.num_stations
        self.max_req_slots = max_req_slots
        self._geom = env._geom

        self.x_span = float(env.x_range[1] - env.x_range[0]) if env.x_range[1] > env.x_range[0] else 1.0
        self.y_span = float(env.y_range[1] - env.y_range[0]) if env.y_range[1] > env.y_range[0] else 1.0
        self.max_time = 86400.0
        self.max_dist = self.x_span + self.y_span

        self.critical_battery = float(env._params.get('Critical_battery_level', {}).get('level', 20.0))
        self.non_critical_battery = float(env._params.get('Non_critical_battery_level', {}).get('level', 80.0))

        # Station info
        stations_df = self._geom.stations
        st_types = stations_df['type'].to_numpy()
        self.charging_station_indices = np.where(st_types == 'Charging station')[0]
        self.num_charging_stations = len(self.charging_station_indices)

        # Feature dimensions:
        # Global: [curr_time / max_time] -> 1 dim
        # Vehicles: V * [x, y, battery, is_busy, is_charging, has_cap] -> V * 6 dims
        # Requests: max_slots * [ox, oy, dx, dy, dist, rel_t, deadline, slack, is_active] -> K * 9 dims
        self.obs_dim = 1 + (self.num_vehicles * 6) + (self.max_req_slots * 9)

    def encode(self, obs: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Converts observation to flat state tensor, request assignment mask, and repositioning mask.
        Returns:
            flat_obs (torch.Tensor): Shape (obs_dim,)
            req_mask (torch.Tensor): Shape (max_req_slots, V + 1)
            repos_mask (torch.Tensor): Shape (V, num_charging_stations + 1)
            num_actual_pending (int): Number of actual active requests (0 <= R <= max_req_slots)
            total_pending (int): Total pending requests in observation (R_total)
        """
        curr_time = float(obs["time"][0])
        v_locs = obs["v_locs"]  # (V, 2)
        v_battery = obs["v_battery"]  # (V,)
        v_jobs = obs["v_jobs"]  # (V, 3)
        V = len(v_locs)

        # 1. Global features
        global_feat = [curr_time / self.max_time]

        # 2. Vehicle features
        norm_v_locs = v_locs / np.array([self.x_span, self.y_span])
        norm_v_bat = (v_battery / 100.0)[:, None]
        is_busy = np.isin(v_jobs[:, 0], (Jobs.SETUP, Jobs.PROCESS)).astype(np.float32)[:, None]
        is_charging = (v_jobs[:, 0] == Jobs.CHARGING).astype(np.float32)[:, None]
        has_capacity = (v_jobs[:, 1] == Jobs.NULL).astype(np.float32)[:, None]

        v_feats = np.hstack([norm_v_locs, norm_v_bat, is_busy, is_charging, has_capacity]).flatten()

        # 3. Request features (Fixed slots with padding)
        req_times = obs["request_times"]
        total_pending = len(req_times)
        num_actual = min(total_pending, self.max_req_slots)

        req_slot_feats = np.zeros((self.max_req_slots, 9), dtype=np.float32)
        req_mask = torch.zeros((self.max_req_slots, V + 1), dtype=torch.bool, device=device)

        if total_pending > 0:
            sort_idx = np.argsort(req_times)[:self.max_req_slots]
            req_locs = obs["request_locs"][sort_idx]
            r_times = req_times[sort_idx]
            r_deadlines = obs["request_deadlines"][sort_idx]

            orig_coords = req_locs[:, 0, :]
            dest_coords = req_locs[:, 1, :]
            norm_orig = orig_coords / np.array([self.x_span, self.y_span])
            norm_dest = dest_coords / np.array([self.x_span, self.y_span])
            trip_dists = np.abs(orig_coords - dest_coords).sum(axis=1, keepdims=True) / self.max_dist
            norm_rel = (r_times[:, None] / self.max_time)
            norm_dead = (r_deadlines[:, None] / self.max_time)
            slack = np.maximum(0.0, r_deadlines - curr_time - (trip_dists.squeeze(-1) * self.max_dist))[:, None] / 3600.0
            is_active = np.ones((len(sort_idx), 1), dtype=np.float32)

            filled_feats = np.hstack([norm_orig, norm_dest, trip_dists, norm_rel, norm_dead, slack, is_active])
            req_slot_feats[:len(sort_idx)] = filled_feats

            for r in range(len(sort_idx)):
                req_mask[r, V] = True  # "Unassigned" option is always valid
                for v in range(V):
                    is_crit = v_battery[v] <= self.critical_battery
                    queue_full = v_jobs[v, 1] != Jobs.NULL
                    if not is_crit and not queue_full:
                        req_mask[r, v] = True

        for r in range(num_actual, self.max_req_slots):
            req_mask[r, V] = True

        req_flat = req_slot_feats.flatten()

        # 4. Repositioning mask: (V, num_c + 1)
        repos_mask = torch.ones((V, self.num_charging_stations + 1), dtype=torch.bool, device=device)
        for v in range(V):
            # Vehicle cannot reposition if it is already in transit (SETUP, PROCESS, REPOSITION), has a queued job, or is stranded (0% battery)
            if v_jobs[v, 0] in (Jobs.SETUP, Jobs.PROCESS, Jobs.REPOSITION) or v_jobs[v, 1] != Jobs.NULL or v_battery[v] <= 1e-4:
                repos_mask[v, :self.num_charging_stations] = False

        flat_obs_np = np.concatenate([global_feat, v_feats, req_flat]).astype(np.float32)
        flat_obs = torch.from_numpy(flat_obs_np).to(device)

        return flat_obs, req_mask, repos_mask, num_actual, total_pending


class DQNNetwork(nn.Module):
    """
    Multi-Head Deep Q-Network for AGV-Gym.
    Supports both Standard DQN and Dueling DQN architecture for joint Request Assignment
    and Repositioning value estimation.
    """

    def __init__(
        self,
        obs_dim: int,
        num_vehicles: int = 9,
        max_req_slots: int = 15,
        num_charging_stations: int = 9,
        hidden_dim: int = 256,
        dueling: bool = True
    ):
        super(DQNNetwork, self).__init__()
        self.num_vehicles = num_vehicles
        self.max_req_slots = max_req_slots
        self.num_charging_stations = num_charging_stations
        self.dueling = dueling

        # Shared MLP Backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if self.dueling:
            # Value Stream: scalar V(s)
            self.val_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            # Advantage Stream 1: Request Assignment A_assign(s, r, v)
            self.adv_assign_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, max_req_slots * (num_vehicles + 1))
            )
            # Advantage Stream 2: Repositioning A_repos(s, v, c)
            self.adv_repos_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_vehicles * (num_charging_stations + 1))
            )
        else:
            # Standard Q-Head for Request Assignment: Q_assign(s, r, v)
            self.q_assign_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, max_req_slots * (num_vehicles + 1))
            )
            # Standard Q-Head for Repositioning: Q_repos(s, v, c)
            self.q_repos_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_vehicles * (num_charging_stations + 1))
            )

    def forward(
        self,
        flat_obs: torch.Tensor,
        req_mask: Optional[torch.Tensor] = None,
        repos_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Q-values for Request Assignment and Repositioning.
        Returns:
            q_assign (torch.Tensor): Shape (..., max_req_slots, num_vehicles + 1)
            q_repos (torch.Tensor): Shape (..., num_vehicles, num_charging_stations + 1)
        """
        is_batched = flat_obs.dim() > 1
        hidden = self.backbone(flat_obs)

        if self.dueling:
            val = self.val_head(hidden)  # (..., 1)
            adv_assign = self.adv_assign_head(hidden)
            adv_repos = self.adv_repos_head(hidden)

            if is_batched:
                batch_size = flat_obs.size(0)
                adv_assign = adv_assign.view(batch_size, self.max_req_slots, self.num_vehicles + 1)
                adv_repos = adv_repos.view(batch_size, self.num_vehicles, self.num_charging_stations + 1)
                val_assign = val.unsqueeze(-1)  # (batch_size, 1, 1)
                val_repos = val.unsqueeze(-1)   # (batch_size, 1, 1)
            else:
                adv_assign = adv_assign.view(self.max_req_slots, self.num_vehicles + 1)
                adv_repos = adv_repos.view(self.num_vehicles, self.num_charging_stations + 1)
                val_assign = val.unsqueeze(-1)
                val_repos = val.unsqueeze(-1)

            # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            q_assign = val_assign + (adv_assign - adv_assign.mean(dim=-1, keepdim=True))
            q_repos = val_repos + (adv_repos - adv_repos.mean(dim=-1, keepdim=True))
        else:
            raw_assign = self.q_assign_head(hidden)
            raw_repos = self.q_repos_head(hidden)

            if is_batched:
                batch_size = flat_obs.size(0)
                q_assign = raw_assign.view(batch_size, self.max_req_slots, self.num_vehicles + 1)
                q_repos = raw_repos.view(batch_size, self.num_vehicles, self.num_charging_stations + 1)
            else:
                q_assign = raw_assign.view(self.max_req_slots, self.num_vehicles + 1)
                q_repos = raw_repos.view(self.num_vehicles, self.num_charging_stations + 1)

        if req_mask is not None:
            q_assign = torch.where(req_mask, q_assign, torch.tensor(-1e8, device=flat_obs.device))
        if repos_mask is not None:
            q_repos = torch.where(repos_mask, q_repos, torch.tensor(-1e8, device=flat_obs.device))

        return q_assign, q_repos


class ReplayBuffer:
    """
    Experience Replay Buffer for Deep Q-Network.
    Stores multi-component transition tuples with action masks for valid replay.
    """

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        flat_obs: torch.Tensor,
        req_mask: torch.Tensor,
        repos_mask: torch.Tensor,
        num_actual: int,
        req_assgts: torch.Tensor,
        repos_raw: torch.Tensor,
        reward: float,
        next_flat_obs: torch.Tensor,
        next_req_mask: torch.Tensor,
        next_repos_mask: torch.Tensor,
        next_num_actual: int,
        done: bool
    ):
        transition = (
            flat_obs.detach().cpu(),
            req_mask.detach().cpu(),
            repos_mask.detach().cpu(),
            num_actual,
            req_assgts.detach().cpu(),
            repos_raw.detach().cpu(),
            reward,
            next_flat_obs.detach().cpu(),
            next_req_mask.detach().cpu(),
            next_repos_mask.detach().cpu(),
            next_num_actual,
            done
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """Samples a random mini-batch of transitions and moves to device."""
        samples = random.sample(self.buffer, batch_size)

        flat_obs_list, req_mask_list, repos_mask_list = [], [], []
        num_actual_list = []
        req_assgts_list, repos_raw_list = [], []
        rewards_list, next_obs_list, next_req_mask_list, next_repos_mask_list = [], [], [], []
        next_num_actual_list = []
        dones_list = []

        for s in samples:
            flat_obs_list.append(s[0])
            req_mask_list.append(s[1])
            repos_mask_list.append(s[2])
            num_actual_list.append(s[3])
            req_assgts_list.append(s[4])
            repos_raw_list.append(s[5])
            rewards_list.append(s[6])
            next_obs_list.append(s[7])
            next_req_mask_list.append(s[8])
            next_repos_mask_list.append(s[9])
            next_num_actual_list.append(s[10])
            dones_list.append(s[11])

        return {
            "flat_obs": torch.stack(flat_obs_list).to(device),
            "req_mask": torch.stack(req_mask_list).to(device),
            "repos_mask": torch.stack(repos_mask_list).to(device),
            "num_actual": num_actual_list,
            "req_assgts": torch.stack(req_assgts_list).to(device),
            "repos_raw": torch.stack(repos_raw_list).to(device),
            "rewards": torch.tensor(rewards_list, dtype=torch.float32, device=device),
            "next_flat_obs": torch.stack(next_obs_list).to(device),
            "next_req_mask": torch.stack(next_req_mask_list).to(device),
            "next_repos_mask": torch.stack(next_repos_mask_list).to(device),
            "next_num_actual": next_num_actual_list,
            "dones": torch.tensor(dones_list, dtype=torch.float32, device=device)
        }


class DQNTrainer:
    """
    Deep Q-Network (DQN) Trainer with Double DQN, Dueling Architecture, Valid Action Masking,
    and TensorBoard Monitoring for AGV-Gym.
    """

    def __init__(
        self,
        env: AGVEnv,
        vec_env: Optional[SubprocVecAGVEnv] = None,
        max_req_slots: int = 15,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9999,
        buffer_capacity: int = 500000,
        batch_size: int = 256,
        utd_ratio: int = 1,
        dueling: bool = True,
        double_dqn: bool = True,
        max_grad_norm: float = 1.0,
        tensorboard: bool = False,
        log_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.env = env
        self.vec_env = vec_env
        self.utd_ratio = utd_ratio
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        worker_desc = f"{vec_env.num_envs} Parallel CPU Workers" if vec_env is not None else "Single Env Worker"
        print(f"DQNTrainer initialized on device: {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}) ({worker_desc})")

        self.encoder = DQNObsEncoder(env, max_req_slots=max_req_slots)
        self.c_indices_np = self.encoder.charging_station_indices
        self.num_vehicles = env.num_vehicles
        self.max_req_slots = max_req_slots
        self.num_charging_stations = self.encoder.num_charging_stations

        # Q-Networks (Online & Target)
        self.dueling = dueling
        self.double_dqn = double_dqn

        self.q_net = DQNNetwork(
            obs_dim=self.encoder.obs_dim,
            num_vehicles=self.num_vehicles,
            max_req_slots=max_req_slots,
            num_charging_stations=self.num_charging_stations,
            hidden_dim=hidden_dim,
            dueling=dueling
        ).to(self.device)

        self.target_net = DQNNetwork(
            obs_dim=self.encoder.obs_dim,
            num_vehicles=self.num_vehicles,
            max_req_slots=max_req_slots,
            num_charging_stations=self.num_charging_stations,
            hidden_dim=hidden_dim,
            dueling=dueling
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.best_eval_cost = float("inf")
        self.best_eval_completion_pct = 0.0

        # TensorBoard setup
        self.tensorboard_enabled = tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.tensorboard_enabled:
            log_directory = log_dir if log_dir else f"runs/dqn_{int(time.time())}"
            os.makedirs(log_directory, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_directory)
            print(f"TensorBoard logging active at: {log_directory}")
        elif tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but torch.utils.tensorboard is not available.")

        self.total_env_steps = 0
        self.total_train_updates = 0

    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False,
        eval_stats: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Saves full DQN checkpoint (Q-network, target network, optimizer, epsilon, hyperparameters).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "format_version": "1.0",
            "timestamp": time.time(),
            "algorithm": "DQN",
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_env_steps": self.total_env_steps,
            "total_train_updates": self.total_train_updates,
            "model_config": {
                "obs_dim": self.encoder.obs_dim,
                "num_vehicles": self.num_vehicles,
                "max_req_slots": self.max_req_slots,
                "num_charging_stations": self.num_charging_stations,
                "hidden_dim": self.hidden_dim,
                "dueling": self.dueling,
                "double_dqn": self.double_dqn,
            },
            "hyperparameters": {
                "lr": self.lr,
                "gamma": self.gamma,
                "tau": self.tau,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "max_grad_norm": self.max_grad_norm,
                "fleet_size": self.env.num_vehicles,
                "num_requests": self.env.num_requests,
                "alpha": self.env.alpha,
            },
            "best_eval_cost": self.best_eval_cost,
            "best_eval_completion_pct": self.best_eval_completion_pct,
            "eval_stats": eval_stats or {}
        }
        torch.save(checkpoint, filepath)
        print(f"  [Checkpoint] Saved DQN checkpoint to: {filepath}")
        return filepath

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Loads a DQN checkpoint (.ckpt or .pt).
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if "q_net_state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            if "target_net_state_dict" in checkpoint:
                self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            else:
                self.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["state_dict"])
            self.target_net.load_state_dict(checkpoint["state_dict"])
        else:
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)

        if load_optimizer:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
            if "total_env_steps" in checkpoint:
                self.total_env_steps = checkpoint["total_env_steps"]
            if "total_train_updates" in checkpoint:
                self.total_train_updates = checkpoint["total_train_updates"]
            if "best_eval_cost" in checkpoint:
                self.best_eval_cost = checkpoint["best_eval_cost"]
            if "best_eval_completion_pct" in checkpoint:
                self.best_eval_completion_pct = checkpoint["best_eval_completion_pct"]

        print(f"  [Checkpoint] Successfully loaded DQN checkpoint from: {filepath} (Total Env Steps: {self.total_env_steps}, Epsilon: {self.epsilon:.4f})")
        return checkpoint

    def select_action(
        self,
        flat_obs: torch.Tensor,
        req_mask: torch.Tensor,
        repos_mask: torch.Tensor,
        num_actual_reqs: int,
        total_pending_reqs: int,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Selects actions with epsilon-greedy exploration and sequential vehicle exclusivity.
        Pads action arrays to match total pending requests.
        """
        V = self.num_vehicles
        K = self.max_req_slots
        num_c = self.num_charging_stations
        device = flat_obs.device

        with torch.no_grad():
            q_assign, q_repos = self.q_net(flat_obs, req_mask, repos_mask)

        req_assgts_list = []
        assigned_v_indices = set()

        # 1. Request Assignment Decision (Slot by slot)
        for r in range(K):
            mask_r = req_mask[r].clone()
            if len(assigned_v_indices) > 0:
                excl_tensor = torch.tensor(list(assigned_v_indices), dtype=torch.long, device=device)
                mask_r[excl_tensor] = False

            if not deterministic and random.random() < epsilon:
                # Epsilon exploration: random valid vehicle
                valid_actions = torch.where(mask_r)[0]
                if len(valid_actions) > 0:
                    chosen_idx = random.choice(valid_actions.tolist())
                    chosen_v = torch.tensor(chosen_idx, dtype=torch.long, device=device)
                else:
                    chosen_v = torch.tensor(V, dtype=torch.long, device=device)
            else:
                # Greedy selection
                masked_q_r = torch.where(mask_r, q_assign[r], torch.tensor(-1e8, device=device))
                chosen_v = torch.argmax(masked_q_r, dim=-1)

            req_assgts_list.append(chosen_v)
            chosen_v_idx = chosen_v.item()
            if chosen_v_idx < V:
                assigned_v_indices.add(chosen_v_idx)

        req_assgts_tensor = torch.stack(req_assgts_list)

        # Pad or slice request assignments to match actual total pending requests in the environment
        req_assgts_np = req_assgts_tensor.cpu().numpy()
        full_req_assgts = np.full(total_pending_reqs, fill_value=V, dtype=int)
        valid_count = min(num_actual_reqs, total_pending_reqs, K)
        full_req_assgts[:valid_count] = req_assgts_np[:valid_count]

        # 2. Idle Repositioning Decision (Vehicle by vehicle)
        repos_raw_list = []
        for v in range(V):
            mask_v = repos_mask[v]
            if not deterministic and random.random() < epsilon:
                valid_repos = torch.where(mask_v)[0]
                if len(valid_repos) > 0:
                    chosen_repos = torch.tensor(random.choice(valid_repos.tolist()), dtype=torch.long, device=device)
                else:
                    chosen_repos = torch.tensor(num_c, dtype=torch.long, device=device)
            else:
                masked_q_repos_v = torch.where(mask_v, q_repos[v], torch.tensor(-1e8, device=device))
                chosen_repos = torch.argmax(masked_q_repos_v, dim=-1)

            repos_raw_list.append(chosen_repos)

        repos_raw_tensor = torch.stack(repos_raw_list)

        # Map repositioning indices: 0..num_c-1 -> charging station IDs, num_c -> stay (78)
        repos_raw_np = repos_raw_tensor.cpu().numpy()
        reposition = np.full(V, fill_value=78, dtype=int)
        for v in range(V):
            choice = repos_raw_np[v]
            if choice < num_c:
                reposition[v] = self.c_indices_np[choice]
            else:
                reposition[v] = 78

        action_dict = {
            "req_rejections": np.zeros(total_pending_reqs, dtype=int),
            "req_assgts": full_req_assgts,
            "reposition": reposition
        }

        return {
            "action_dict": action_dict,
            "req_assgts_tensor": req_assgts_tensor,
            "repos_raw_tensor": repos_raw_tensor,
            "q_assign": q_assign,
            "q_repos": q_repos
        }

    def select_actions_batch(
        self,
        flat_obs_batch: torch.Tensor,
        req_mask_batch: torch.Tensor,
        repos_mask_batch: torch.Tensor,
        num_actual_list: List[int],
        total_pending_list: List[int],
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
        """
        Performs 1 single batched CUDA forward pass for all N workers,
        followed by sequential exclusivity decoding.
        """
        N = flat_obs_batch.size(0)
        V = self.num_vehicles
        K = self.max_req_slots
        num_c = self.num_charging_stations
        device = flat_obs_batch.device

        with torch.no_grad():
            q_assign_batch, q_repos_batch = self.q_net(flat_obs_batch, req_mask_batch, repos_mask_batch)

        actions_list = []
        req_assgts_batch = []
        repos_raw_batch = []

        for i in range(N):
            q_assign = q_assign_batch[i]  # (K, V+1)
            q_repos = q_repos_batch[i]    # (V, num_c+1)
            req_mask = req_mask_batch[i]
            repos_mask = repos_mask_batch[i]
            num_actual = num_actual_list[i]
            total_pending = total_pending_list[i]

            req_assgts_list = []
            assigned_v_indices = set()

            for r in range(K):
                mask_r = req_mask[r].clone()
                if len(assigned_v_indices) > 0:
                    excl_tensor = torch.tensor(list(assigned_v_indices), dtype=torch.long, device=device)
                    mask_r[excl_tensor] = False

                if not deterministic and random.random() < epsilon:
                    valid_actions = torch.where(mask_r)[0]
                    if len(valid_actions) > 0:
                        chosen_idx = random.choice(valid_actions.tolist())
                        chosen_v = torch.tensor(chosen_idx, dtype=torch.long, device=device)
                    else:
                        chosen_v = torch.tensor(V, dtype=torch.long, device=device)
                else:
                    masked_q_r = torch.where(mask_r, q_assign[r], torch.tensor(-1e8, device=device))
                    chosen_v = torch.argmax(masked_q_r, dim=-1)

                req_assgts_list.append(chosen_v)
                chosen_v_idx = chosen_v.item()
                if chosen_v_idx < V:
                    assigned_v_indices.add(chosen_v_idx)

            req_assgts_tensor = torch.stack(req_assgts_list)
            req_assgts_batch.append(req_assgts_tensor)

            req_assgts_np = req_assgts_tensor.cpu().numpy()
            full_req_assgts = np.full(total_pending, fill_value=V, dtype=int)
            valid_count = min(num_actual, total_pending, K)
            full_req_assgts[:valid_count] = req_assgts_np[:valid_count]

            repos_raw_list = []
            for v in range(V):
                mask_v = repos_mask[v]
                if not deterministic and random.random() < epsilon:
                    valid_repos = torch.where(mask_v)[0]
                    if len(valid_repos) > 0:
                        chosen_repos = torch.tensor(random.choice(valid_repos.tolist()), dtype=torch.long, device=device)
                    else:
                        chosen_repos = torch.tensor(num_c, dtype=torch.long, device=device)
                else:
                    masked_q_repos_v = torch.where(mask_v, q_repos[v], torch.tensor(-1e8, device=device))
                    chosen_repos = torch.argmax(masked_q_repos_v, dim=-1)
                repos_raw_list.append(chosen_repos)

            repos_raw_tensor = torch.stack(repos_raw_list)
            repos_raw_batch.append(repos_raw_tensor)

            repos_raw_np = repos_raw_tensor.cpu().numpy()
            reposition = np.full(V, fill_value=78, dtype=int)
            for v in range(V):
                choice = repos_raw_np[v]
                if choice < num_c:
                    reposition[v] = self.c_indices_np[choice]
                else:
                    reposition[v] = 78

            actions_list.append({
                "req_rejections": np.zeros(total_pending, dtype=int),
                "req_assgts": full_req_assgts,
                "reposition": reposition
            })

        return actions_list, torch.stack(req_assgts_batch), torch.stack(repos_raw_batch)

    def train_step(self) -> Dict[str, float]:
        """
        Performs one DQN gradient update step from the replay buffer.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        self.q_net.train()
        batch = self.buffer.sample(self.batch_size, self.device)

        flat_obs = batch["flat_obs"]          # (B, obs_dim)
        req_mask = batch["req_mask"]          # (B, K, V+1)
        repos_mask = batch["repos_mask"]      # (B, V, num_c+1)
        req_assgts = batch["req_assgts"]      # (B, K)
        repos_raw = batch["repos_raw"]        # (B, V)
        rewards = batch["rewards"]            # (B,)
        next_flat_obs = batch["next_flat_obs"]
        next_req_mask = batch["next_req_mask"]
        next_repos_mask = batch["next_repos_mask"]
        dones = batch["dones"]                # (B,)
        num_actuals = batch["num_actual"]

        B = flat_obs.size(0)
        K = self.max_req_slots
        V = self.num_vehicles
        num_c = self.num_charging_stations

        # 1. Evaluate Predicted Q-values for taken actions
        q_assign_pred, q_repos_pred = self.q_net(flat_obs, req_mask, repos_mask)
        q_assign_taken = q_assign_pred.gather(dim=-1, index=req_assgts.unsqueeze(-1)).squeeze(-1)  # (B, K)
        q_repos_taken = q_repos_pred.gather(dim=-1, index=repos_raw.unsqueeze(-1)).squeeze(-1)    # (B, V)

        # Average action Q over valid slots and vehicles
        req_active_mask = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        for i, n_act in enumerate(num_actuals):
            req_active_mask[i, :min(n_act, K)] = True

        q_assign_mean = (q_assign_taken * req_active_mask.float()).sum(dim=-1) / (req_active_mask.float().sum(dim=-1) + 1e-8)
        q_repos_mean = q_repos_taken.mean(dim=-1)

        joint_pred_q = 0.5 * (q_assign_mean + q_repos_mean)  # (B,)

        # 2. Compute Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selected by online net, evaluated by target net
                next_q_assign_online, next_q_repos_online = self.q_net(next_flat_obs, next_req_mask, next_repos_mask)
                next_q_assign_target, next_q_repos_target = self.target_net(next_flat_obs, next_req_mask, next_repos_mask)

                # Greedy actions under online net with validity mask
                masked_next_q_assign = torch.where(next_req_mask, next_q_assign_online, torch.tensor(-1e8, device=self.device))
                next_best_assign = torch.argmax(masked_next_q_assign, dim=-1, keepdim=True)
                next_max_q_assign = next_q_assign_target.gather(dim=-1, index=next_best_assign).squeeze(-1)

                masked_next_q_repos = torch.where(next_repos_mask, next_q_repos_online, torch.tensor(-1e8, device=self.device))
                next_best_repos = torch.argmax(masked_next_q_repos, dim=-1, keepdim=True)
                next_max_q_repos = next_q_repos_target.gather(dim=-1, index=next_best_repos).squeeze(-1)
            else:
                # Standard Target DQN
                next_q_assign_target, next_q_repos_target = self.target_net(next_flat_obs, next_req_mask, next_repos_mask)
                masked_next_q_assign = torch.where(next_req_mask, next_q_assign_target, torch.tensor(-1e8, device=self.device))
                next_max_q_assign = torch.max(masked_next_q_assign, dim=-1)[0]

                masked_next_q_repos = torch.where(next_repos_mask, next_q_repos_target, torch.tensor(-1e8, device=self.device))
                next_max_q_repos = torch.max(masked_next_q_repos, dim=-1)[0]

            next_joint_target = 0.5 * (next_max_q_assign.mean(dim=-1) + next_max_q_repos.mean(dim=-1))
            td_target = rewards + (1.0 - dones.float()) * self.gamma * next_joint_target

        # 3. Compute Huber / Smooth L1 Loss
        loss = F.smooth_l1_loss(joint_pred_q, td_target)

        # 4. Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 5. Soft Update Target Network
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # 6. Epsilon Decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.total_train_updates += 1

        loss_val = loss.item()
        q_mean_val = joint_pred_q.mean().item()
        td_err_val = torch.abs(joint_pred_q - td_target).mean().item()

        if self.writer is not None and self.total_train_updates % 10 == 0:
            self.writer.add_scalar("Loss/total_loss", loss_val, self.total_train_updates)
            self.writer.add_scalar("Loss/q_joint_mean", q_mean_val, self.total_train_updates)
            self.writer.add_scalar("Loss/td_error", td_err_val, self.total_train_updates)
            self.writer.add_scalar("Epsilon/exploration_rate", self.epsilon, self.total_train_updates)

        return {
            "loss": loss_val,
            "q_mean": q_mean_val,
            "td_error": td_err_val,
            "epsilon": self.epsilon
        }

    def train(
        self,
        total_steps: Optional[int] = None,
        total_episodes: Optional[int] = None,
        warmup_steps: int = 128,
        train_freq: int = 4,
        save_dir: Optional[str] = None,
        save_freq: int = 0,
        render: bool = False,
        render_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Executes online DQN training loop over a given number of environment steps or full episodes.
        Supports both single-worker and multi-worker vectorized simulation.
        """
        target_episodes = total_episodes if (total_episodes is not None and total_episodes > 0) else None
        target_steps = total_steps if (total_steps is not None and total_steps > 0) else None

        if target_episodes is None and target_steps is None:
            target_steps = 1000  # Default fallback

        # --- Multi-Worker Vectorized Training Loop ---
        if self.vec_env is not None:
            num_envs = self.vec_env.num_envs
            obs_list = self.vec_env.reset()
            episodes_completed = 0
            ep_rewards = [0.0] * num_envs
            losses = []
            step = 0
            prev_step = 0
            start_time = time.time()

            while True:
                # 1. Encode observations for all parallel workers
                flat_list, req_m_list, repos_m_list = [], [], []
                num_actual_list, total_pending_list = [], []
                for o in obs_list:
                    f_obs, r_mask, rep_mask, n_act, tot_p = self.encoder.encode(o, self.device)
                    flat_list.append(f_obs)
                    req_m_list.append(r_mask)
                    repos_m_list.append(rep_mask)
                    num_actual_list.append(n_act)
                    total_pending_list.append(tot_p)

                flat_batch = torch.stack(flat_list)
                req_m_batch = torch.stack(req_m_list)
                repos_m_batch = torch.stack(repos_m_list)

                # 2. Batched CUDA forward pass and action selection
                eps = 1.0 if step <= warmup_steps else self.epsilon
                actions_list, req_assgts_batch, repos_raw_batch = self.select_actions_batch(
                    flat_batch, req_m_batch, repos_m_batch, num_actual_list, total_pending_list,
                    epsilon=eps, deterministic=False
                )

                # 3. Concurrent step across all workers
                next_obs_list, rewards, dones, infos = self.vec_env.step(actions_list)

                # 4. Encode next observations
                next_flat_list, next_req_m_list, next_repos_m_list = [], [], []
                next_num_act_list = []
                for no in next_obs_list:
                    nf_obs, nr_mask, nrep_mask, nn_act, _ = self.encoder.encode(no, self.device)
                    next_flat_list.append(nf_obs)
                    next_req_m_list.append(nr_mask)
                    next_repos_m_list.append(nrep_mask)
                    next_num_act_list.append(nn_act)

                # 5. Push all N transitions to centralized replay buffer
                for i in range(num_envs):
                    self.buffer.add(
                        flat_obs=flat_list[i],
                        req_mask=req_m_list[i],
                        repos_mask=repos_m_list[i],
                        num_actual=num_actual_list[i],
                        req_assgts=req_assgts_batch[i],
                        repos_raw=repos_raw_batch[i],
                        reward=float(rewards[i]),
                        next_flat_obs=next_flat_list[i],
                        next_req_mask=next_req_m_list[i],
                        next_repos_mask=next_repos_m_list[i],
                        next_num_actual=next_num_act_list[i],
                        done=bool(dones[i])
                    )
                    ep_rewards[i] += float(rewards[i])

                prev_step = step
                step += num_envs
                self.total_env_steps += num_envs

                # 6. High-Throughput GPU Gradient Updates
                if step > warmup_steps:
                    for _ in range(self.utd_ratio):
                        train_info = self.train_step()
                        if "loss" in train_info:
                            losses.append(train_info["loss"])

                if self.writer is not None:
                    self.writer.add_scalar("Rollout/mean_step_reward", float(np.mean(rewards)), self.total_env_steps)

                # 7. Periodic step-based checkpoint saving
                if save_dir and save_freq > 0 and (step // save_freq > prev_step // save_freq):
                    self.save_checkpoint(os.path.join(save_dir, f"dqn_step_{step}.ckpt"))

                # 8. Check completed worker episodes
                for i in range(num_envs):
                    if dones[i]:
                        episodes_completed += 1
                        ep_r = ep_rewards[i]
                        ep_rewards[i] = 0.0

                        term_info = infos[i].get("terminal_info", {})
                        total_cost = term_info.get("total_cost", 0.0)
                        tardiness_cost = term_info.get("tardiness_cost", 0.0)
                        travel_dist = term_info.get("travel_distance", 0.0)
                        completed_reqs = term_info.get("completed_requests", 0)
                        stranded_count = term_info.get("stranded_agvs", 0)
                        total_reqs = max(1, term_info.get("num_requests", self.env.num_requests))
                        comp_pct = min(100.0, (completed_reqs / total_reqs) * 100.0)
                        num_v = term_info.get("num_vehicles", self.env.num_vehicles)

                        print(f"  [Train Episode {episodes_completed}{f'/{target_episodes}' if target_episodes else ''} (Worker {i})] "
                              f"Steps: {step} | Reward: {ep_r:.2f} | "
                              f"Total Cost: {total_cost:.2f} (Tardiness: {tardiness_cost:.2f}, Travel: {travel_dist:.2f}) | "
                              f"Completed Reqs: {completed_reqs}/{total_reqs} ({comp_pct:.1f}%) | "
                              f"Stranded: {stranded_count}/{num_v} | "
                              f"Epsilon: {self.epsilon:.4f}", flush=True)

                        if self.writer is not None:
                            self.writer.add_scalar("Rollout/episode_reward", ep_r, episodes_completed)
                            self.writer.add_scalar("Rollout/episode_cost", total_cost, episodes_completed)
                            self.writer.add_scalar("Rollout/travel_distance", travel_dist, episodes_completed)
                            self.writer.add_scalar("Rollout/tardiness_cost", tardiness_cost, episodes_completed)
                            self.writer.add_scalar("Rollout/completed_requests", completed_reqs, episodes_completed)
                            self.writer.add_scalar("Rollout/completion_rate_pct", comp_pct, episodes_completed)
                            self.writer.add_scalar("Rollout/stranded_agvs", stranded_count, episodes_completed)

                        if save_dir and save_freq > 0 and episodes_completed % save_freq == 0 and target_episodes is not None:
                            self.save_checkpoint(os.path.join(save_dir, f"dqn_ep_{episodes_completed}.ckpt"))

                obs_list = next_obs_list

                if target_episodes is not None and episodes_completed >= target_episodes:
                    break
                if target_steps is not None and step >= target_steps:
                    break

            if save_dir:
                self.save_checkpoint(os.path.join(save_dir, "dqn_latest.ckpt"))

            elapsed = time.time() - start_time
            return {
                "total_steps": step,
                "episodes_completed": episodes_completed,
                "mean_loss": float(np.mean(losses)) if losses else 0.0,
                "elapsed_time": elapsed,
                "epsilon": self.epsilon
            }

        # --- Single-Worker Training Loop ---
        obs = self.env.reset()
        episodes_completed = 0
        ep_rewards = []
        curr_ep_reward = 0.0
        losses = []
        step = 0

        if render:
            plt.figure("AGV-Gym DQN Training", figsize=(12, 7))
            rgb = self.env.render()
            im = self.env.draw_label(rgb, episodes_completed, 0.0)
            plt.imshow(im)
            plt.ion()
            plt.show()

        start_time = time.time()
        while True:
            step += 1
            self.total_env_steps += 1
            flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)

            eps = 1.0 if step <= warmup_steps else self.epsilon
            act_out = self.select_action(
                flat_obs, req_mask, repos_mask, num_actual, total_pending, epsilon=eps, deterministic=False
            )

            next_obs, reward, terminal, info = self.env.step(act_out["action_dict"])
            curr_ep_reward += reward

            next_flat, next_req_m, next_repos_m, next_num_act, _ = self.encoder.encode(next_obs, self.device)

            self.buffer.add(
                flat_obs=flat_obs,
                req_mask=req_mask,
                repos_mask=repos_mask,
                num_actual=num_actual,
                req_assgts=act_out["req_assgts_tensor"],
                repos_raw=act_out["repos_raw_tensor"],
                reward=reward,
                next_flat_obs=next_flat,
                next_req_mask=next_req_m,
                next_repos_mask=next_repos_m,
                next_num_actual=next_num_act,
                done=terminal
            )

            # Training gradient step
            if step > warmup_steps and step % train_freq == 0:
                for _ in range(self.utd_ratio):
                    train_info = self.train_step()
                    if "loss" in train_info:
                        losses.append(train_info["loss"])

            if self.writer is not None:
                self.writer.add_scalar("Rollout/step_reward", reward, self.total_env_steps)

            if render and (step % render_interval == 0 or terminal):
                rgb = self.env.render()
                im = self.env.draw_label(rgb, episodes_completed, self.env.rewards, info=info)
                plt.clf()
                plt.imshow(im)
                plt.pause(0.001)

            if save_dir and save_freq > 0 and step % save_freq == 0:
                self.save_checkpoint(os.path.join(save_dir, f"dqn_step_{step}.ckpt"))

            if terminal:
                episodes_completed += 1
                ep_rewards.append(curr_ep_reward)

                total_reqs = max(1, len(self.env._requests) - 1)
                comp_pct = min(100.0, (self.env.num_completed_requests / total_reqs) * 100.0)

                stranded_count = self.env.num_stranded_vehicles
                print(f"  [Train Episode {episodes_completed}{f'/{target_episodes}' if target_episodes else ''}] "
                      f"Steps: {step} | Reward: {curr_ep_reward:.2f} | "
                      f"Total Cost: {self.env.total_cost:.2f} (Tardiness: {self.env.total_tardiness_cost:.2f}, Travel: {self.env.total_travel_distance:.2f}) | "
                      f"Completed Reqs: {self.env.num_completed_requests}/{total_reqs} ({comp_pct:.1f}%) | "
                      f"Stranded: {stranded_count}/{self.env._V} | "
                      f"Epsilon: {self.epsilon:.4f}", flush=True)

                if self.writer is not None:
                    self.writer.add_scalar("Rollout/episode_reward", curr_ep_reward, episodes_completed)
                    self.writer.add_scalar("Rollout/episode_cost", self.env.total_cost, episodes_completed)
                    self.writer.add_scalar("Rollout/travel_distance", self.env.total_travel_distance, episodes_completed)
                    self.writer.add_scalar("Rollout/tardiness_cost", self.env.total_tardiness_cost, episodes_completed)
                    self.writer.add_scalar("Rollout/completed_requests", self.env.num_completed_requests, episodes_completed)
                    self.writer.add_scalar("Rollout/completion_rate_pct", comp_pct, episodes_completed)
                    self.writer.add_scalar("Rollout/stranded_agvs", stranded_count, episodes_completed)

                if save_dir and save_freq > 0 and episodes_completed % save_freq == 0 and target_episodes is not None:
                    self.save_checkpoint(os.path.join(save_dir, f"dqn_ep_{episodes_completed}.ckpt"))

                curr_ep_reward = 0.0
                obs = self.env.reset()

                if target_episodes is not None and episodes_completed >= target_episodes:
                    break
            else:
                obs = next_obs

            if target_steps is not None and step >= target_steps:
                break

        if save_dir:
            self.save_checkpoint(os.path.join(save_dir, "dqn_latest.ckpt"))

        elapsed = time.time() - start_time
        return {
            "total_steps": step,
            "episodes_completed": episodes_completed,
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else curr_ep_reward,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "elapsed_time": elapsed,
            "epsilon": self.epsilon
        }

    def evaluate(
        self,
        num_episodes: int = 1,
        save_dir: Optional[str] = None,
        render: bool = False,
        render_interval: int = 5,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluates the current DQN policy over multiple episodes.

        Returns:
            Dict containing mean cost, travel distance, tardiness, completed requests, stranded AGVs, and reward.
        """
        self.q_net.eval()
        eps_costs = []
        eps_dists = []
        eps_tards = []
        eps_completed = []
        eps_stranded = []
        eps_rewards = []

        total_reqs = max(1, self.env.num_requests)

        for ep in range(num_episodes):
            obs = self.env.reset()
            total_reqs = max(1, len(self.env._requests) - 1)
            terminal = False
            step_count = 0
            start_t = time.time()

            if render:
                plt.figure("AGV-Gym DQN Evaluation", figsize=(12, 7))
                rgb = self.env.render()
                im = self.env.draw_label(rgb, ep + 1, 0.0)
                plt.imshow(im)
                plt.ion()
                plt.show()

            while not terminal:
                flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)
                act_out = self.select_action(
                    flat_obs,
                    req_mask,
                    repos_mask,
                    num_actual,
                    total_pending,
                    epsilon=0.0,
                    deterministic=deterministic
                )

                obs, rwd, terminal, info = self.env.step(act_out["action_dict"])
                step_count += 1

                if step_count % 200 == 0:
                    sim_pct = (self.env.time / 86400.0) * 100.0
                    comp_pct = (self.env.num_completed_requests / max(1, total_reqs)) * 100.0
                    print(f"  [DQN Eval Ep {ep+1}] Step {step_count:4d} | Sim Time: {self.env.time:.0f}s/86400s ({sim_pct:.1f}%) | "
                          f"Completed: {self.env.num_completed_requests}/{total_reqs} ({comp_pct:.1f}%) | "
                          f"Vehicles Moving: {np.sum(np.isin(self.env._vehicles['j1m'], (Jobs.SETUP, Jobs.PROCESS, Jobs.REPOSITION)))}/{self.env._V}",
                          flush=True)

                if render and (step_count % render_interval == 0 or terminal):
                    rgb = self.env.render()
                    im = self.env.draw_label(rgb, ep + 1, self.env.rewards, info=info)
                    plt.clf()
                    plt.imshow(im)
                    plt.pause(0.001)

            elapsed = time.time() - start_t
            comp_pct = min(100.0, (self.env.num_completed_requests / max(1, total_reqs)) * 100.0)
            stranded_count = self.env.num_stranded_vehicles
            stranded_pct = (stranded_count / self.env._V) * 100.0

            print(f"DQN Eval Episode {ep+1}/{num_episodes} completed in {elapsed:.2f}s ({step_count} steps):", flush=True)
            print(f"  - Total Cost: {self.env.total_cost:.2f}", flush=True)
            print(f"  - Tardiness Cost: {self.env.total_tardiness_cost:.2f}", flush=True)
            print(f"  - Travel Distance / Cost: {self.env.total_travel_distance:.2f}", flush=True)
            print(f"  - Completed Requests: {self.env.num_completed_requests}/{total_reqs} ({comp_pct:.1f}%)", flush=True)
            print(f"  - Stranded AGVs (Depleted Battery): {stranded_count}/{self.env._V} ({stranded_pct:.1f}%)", flush=True)

            eps_costs.append(self.env.total_cost)
            eps_dists.append(self.env.total_travel_distance)
            eps_tards.append(self.env.total_tardiness_cost)
            eps_completed.append(self.env.num_completed_requests)
            eps_stranded.append(stranded_count)
            eps_rewards.append(self.env.rewards)

        mean_cost = float(np.mean(eps_costs))
        mean_dist = float(np.mean(eps_dists))
        mean_tard = float(np.mean(eps_tards))
        mean_comp = float(np.mean(eps_completed))
        mean_comp_pct = min(100.0, (mean_comp / max(1, total_reqs)) * 100.0)
        mean_stranded = float(np.mean(eps_stranded))
        mean_stranded_pct = float((mean_stranded / self.env._V) * 100.0)
        mean_rwd = float(np.mean(eps_rewards))

        if self.writer is not None:
            self.writer.add_scalar("Eval/mean_cost", mean_cost, self.total_train_updates)
            self.writer.add_scalar("Eval/mean_dist", mean_dist, self.total_train_updates)
            self.writer.add_scalar("Eval/mean_tardiness", mean_tard, self.total_train_updates)
            self.writer.add_scalar("Eval/mean_completed", mean_comp, self.total_train_updates)
            self.writer.add_scalar("Eval/completion_rate_pct", mean_comp_pct, self.total_train_updates)
            self.writer.add_scalar("Eval/mean_stranded_agvs", mean_stranded, self.total_train_updates)
            self.writer.add_scalar("Eval/stranded_rate_pct", mean_stranded_pct, self.total_train_updates)
            self.writer.add_scalar("Eval/mean_reward", mean_rwd, self.total_train_updates)

        eval_results = {
            "mean_cost": mean_cost,
            "mean_dist": mean_dist,
            "mean_tardiness": mean_tard,
            "mean_completed": mean_comp,
            "completion_rate_pct": mean_comp_pct,
            "mean_stranded": mean_stranded,
            "stranded_rate_pct": mean_stranded_pct,
            "mean_reward": mean_rwd
        }

        if save_dir is not None:
            is_better_completion = mean_comp_pct > self.best_eval_completion_pct
            is_same_comp_lower_cost = (mean_comp_pct == self.best_eval_completion_pct) and (mean_cost < self.best_eval_cost)
            if is_better_completion or is_same_comp_lower_cost:
                self.best_eval_completion_pct = mean_comp_pct
                self.best_eval_cost = mean_cost
                self.save_checkpoint(os.path.join(save_dir, "dqn_best.ckpt"), is_best=True, eval_stats=eval_results)

        return eval_results


def main():
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN) for AGV-Gym")
    parser.add_argument("--render", action="store_true", help="Enable live rendering")
    parser.add_argument("--render_interval", type=int, default=5, help="Render frequency (every N steps)")
    parser.add_argument("--fleet_size", type=int, default=9, help="Fleet size (9, 12, 15, 18)")
    parser.add_argument("--num_requests", "--requests", type=int, default=450, help="Number of requests (450, 900, 1800)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha trade-off coefficient (default: 0.9)")
    parser.add_argument("--train_episodes", type=int, default=0, help="Number of full DQN training episodes (e.g. 5, 10)")
    parser.add_argument("--train_steps", type=int, default=0, help="Number of DQN training steps")
    parser.add_argument("--eval_episodes", "--num_eval_episodes", "--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel CPU simulation workers (e.g. 8 or 16)")
    parser.add_argument("--utd_ratio", type=int, default=1, help="Update-To-Data ratio: gradient updates per parallel step (default: 1)")
    parser.add_argument("--buffer_capacity", type=int, default=500000, help="Replay buffer capacity (default: 500000)")
    parser.add_argument("--warmup_steps", type=int, default=256, help="Number of warmup steps for replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network soft update rate")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.9999, help="Multiplicative epsilon decay rate per update")
    parser.add_argument("--dueling", action="store_true", default=True, help="Use Dueling DQN architecture")
    parser.add_argument("--no_dueling", dest="dueling", action="store_false", help="Disable Dueling architecture")
    parser.add_argument("--double_dqn", action="store_true", default=True, help="Use Double DQN")
    parser.add_argument("--no_double", dest="double_dqn", action="store_false", help="Disable Double DQN")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging")
    parser.add_argument("--no_tensorboard", dest="tensorboard", action="store_false", help="Disable TensorBoard")
    parser.add_argument("--log_dir", type=str, default=None, help="Custom TensorBoard log directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic exploration during evaluation")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints (.ckpt)")
    parser.add_argument("--save_freq", type=int, default=0, help="Interval in steps/episodes to save periodic snapshots (default: 0)")
    parser.add_argument("--load_model", type=str, default=None, help="Path to .ckpt model file to load for evaluation or training")
    parser.add_argument("--resume", type=str, default=None, help="Path to .ckpt file to resume full training state (weights + optimizers + counters)")

    args, unknown = parser.parse_known_args()
    render_val = args.render
    stochastic_eval = args.stochastic
    tensorboard_val = args.tensorboard

    # Default training behavior if neither train_episodes nor train_steps specified
    train_episodes_val = args.train_episodes
    train_steps_val = args.train_steps
    eval_episodes_val = args.eval_episodes

    # Support key=value syntax (e.g. render=true, train_episodes=5, train_steps=500)
    for u in unknown:
        if "=" in u:
            k, v = u.lower().split("=", 1)
            if k in ["render", "--render"]:
                render_val = v in ["true", "1", "yes", "y", "t"]
            elif k in ["fleet_size", "--fleet_size", "num_vehicles"]:
                args.fleet_size = int(v)
            elif k in ["num_requests", "--num_requests", "requests"]:
                args.num_requests = int(v)
            elif k in ["num_workers", "--num_workers"]:
                args.num_workers = int(v)
            elif k in ["batch_size", "--batch_size"]:
                args.batch_size = int(v)
            elif k in ["utd_ratio", "--utd_ratio"]:
                args.utd_ratio = int(v)
            elif k in ["buffer_capacity", "--buffer_capacity"]:
                args.buffer_capacity = int(v)
            elif k in ["alpha", "--alpha"]:
                args.alpha = float(v)
            elif k in ["save_dir", "--save_dir"]:
                args.save_dir = v
            elif k in ["save_freq", "--save_freq"]:
                args.save_freq = int(v)
            elif k in ["load_model", "--load_model"]:
                args.load_model = v
            elif k in ["resume", "--resume"]:
                args.resume = v
            elif k in ["train_episodes", "--train_episodes"]:
                train_episodes_val = int(v)
            elif k in ["train_steps", "--train_steps"]:
                train_steps_val = int(v)
            elif k in ["eval_episodes", "--eval_episodes", "episodes", "--episodes", "num_eval_episodes"]:
                eval_episodes_val = int(v)
            elif k in ["epsilon_start", "--epsilon_start"]:
                args.epsilon_start = float(v)
            elif k in ["epsilon_end", "--epsilon_end"]:
                args.epsilon_end = float(v)
            elif k in ["epsilon_decay", "--epsilon_decay"]:
                args.epsilon_decay = float(v)
            elif k in ["tensorboard", "--tensorboard"]:
                tensorboard_val = v in ["true", "1", "yes", "y", "t"]
            elif k in ["log_dir", "--log_dir"]:
                args.log_dir = v
            elif k in ["device", "--device"]:
                args.device = v
            elif k in ["render_interval", "--render_interval"]:
                args.render_interval = int(v)
            elif k in ["stochastic", "--stochastic"]:
                stochastic_eval = v in ["true", "1", "yes", "y", "t"]
        elif u.lower() in ["render", "render=true"]:
            render_val = True

    # If neither train_steps nor train_episodes was explicitly passed, default to 200 train steps for quick run
    if train_episodes_val == 0 and train_steps_val == 0 and args.load_model is None and args.resume is None:
        train_steps_val = 200

    env_params = {
        "num_vehicles": args.fleet_size,
        "num_requests": args.num_requests,
        "alpha": args.alpha,
        "stochastic": False,
        "seed": 42,
        "max_interdecision_time": 60,
        "for_evaluation": True,
        "nickname": "agv_dqn_eval"
    }

    env = AGVEnv(**env_params)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Multi-worker vectorized environment setup
    vec_env = None
    if args.num_workers > 1:
        factories = [
            EnvFactory(
                num_vehicles=args.fleet_size,
                num_requests=args.num_requests,
                alpha=args.alpha,
                stochastic=False,
                seed=42 + i * 1000,
                max_interdecision_time=60,
                for_evaluation=False,
                nickname=f"agv_dqn_worker_{i}"
            )
            for i in range(args.num_workers)
        ]
        vec_env = SubprocVecAGVEnv(factories)

    trainer = DQNTrainer(
        env=env,
        vec_env=vec_env,
        hidden_dim=256,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        utd_ratio=args.utd_ratio,
        dueling=args.dueling,
        double_dqn=args.double_dqn,
        tensorboard=tensorboard_val,
        log_dir=args.log_dir,
        device=device
    )

    if args.resume:
        trainer.load_checkpoint(args.resume, load_optimizer=True)
    elif args.load_model:
        trainer.load_checkpoint(args.load_model, load_optimizer=False)

    try:
        if train_episodes_val > 0 or train_steps_val > 0:
            mode_desc = f"{train_episodes_val} episodes" if train_episodes_val > 0 else f"{train_steps_val} steps"
            workers_desc = f"Workers={args.num_workers}" if args.num_workers > 1 else "Single Env"
            print(f"\n--- Running DQN Training ({mode_desc}, {workers_desc}, Batch={args.batch_size}, UTD={args.utd_ratio}, alpha={args.alpha}, Dueling={args.dueling}, Double={args.double_dqn}, TensorBoard={tensorboard_val}) ---")
            train_stats = trainer.train(
                total_steps=train_steps_val if train_episodes_val == 0 else None,
                total_episodes=train_episodes_val if train_episodes_val > 0 else None,
                warmup_steps=args.warmup_steps,
                train_freq=4,
                save_dir=args.save_dir,
                save_freq=args.save_freq,
                render=render_val if vec_env is None else False,
                render_interval=args.render_interval
            )
            print(f"DQN Training Completed in {train_stats['elapsed_time']:.2f}s ({train_stats['total_steps']} steps, {train_stats['episodes_completed']} eps) | Mean Loss: {train_stats['mean_loss']:.4f} | Final Epsilon: {train_stats['epsilon']:.4f}")

        if eval_episodes_val > 0:
            print(f"\n--- Evaluating DQN Policy ({eval_episodes_val} episodes, alpha={args.alpha}, deterministic={not stochastic_eval}) ---")
            eval_stats = trainer.evaluate(
                num_episodes=eval_episodes_val,
                save_dir=args.save_dir,
                render=render_val,
                render_interval=max(1, args.render_interval),
                deterministic=not stochastic_eval
            )
            print(f"DQN Evaluation Summary: {eval_stats}")
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
