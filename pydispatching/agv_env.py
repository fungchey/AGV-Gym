from typing import List, Optional, Tuple, Union, Sequence, Dict, Any
import json
import logging
import math
import pkg_resources
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space, MultiBinary, Box, MultiDiscrete, Dict as DictSpace
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
from PIL import Image, ImageFont, ImageDraw

from pydispatching.core import AGVGeometry, Jobs


class InitMultiBinary(Space):
    def __init__(self, n, seed=None):
        if isinstance(n, (Sequence, np.ndarray)):
            self.n = tuple(int(i) for i in n)
        else:
            self.n = int(n)
        super().__init__((self.n,), np.int8, seed)

    def sample(self):
        return self.np_random.integers(low=0, high=2, size=self.n, dtype=self.dtype)

    def contains(self, x):
        return bool(
            isinstance(x, np.ndarray)
            and (x.shape == (self.n,))
            and np.all((x == 0) | (x == 1))
        )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class AGVEnv(gym.Env): 
    """
    AGV-Gym: Deep reinforcement learning environment for battery-constrained
    automated guided vehicles dispatching and scheduling problems (ASP-BC).

    Models the problem as a Constrained Markov Decision Process (CMDP) as described
    in Fung & Wong (2026).
    """

    COMPETITION_SEED = 20151101
    COMPETITION_NUM_EVAL_EPISODES = 50

    def __init__(
        self,
        num_vehicles: Optional[int] = None,
        num_requests: Optional[int] = None,
        request_scenario: Optional[str] = None,
        alpha: Optional[float] = None,
        stochastic: bool = False,
        seed: int = 42,
        action_timelimit: float = np.inf,
        max_interdecision_time: Optional[float] = 60.0,
        for_evaluation: bool = False,
        nickname: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None,
    ):
        """Instantiates the AGV dispatching environment.
        
        Args:
            num_vehicles: Fleet size (9, 12, 15, 18). Default from parameters.json (9).
            num_requests: Total expected requests over horizon (450, 900, 1800).
            request_scenario: "not-busy" (450), "typical" (900), "busy" (1800).
            alpha: Weight for tardiness cost in objective function (0 <= alpha <= 1). Default 0.9.
            stochastic: Whether vehicle movements have stochastic travel times.
            seed: RNG seed for reproducible execution.
            action_timelimit: Maximum wall-clock time in seconds for action return.
            max_interdecision_time: Maximum simulated time interval between decision epochs (s).
            for_evaluation: Evaluation mode flag.
            nickname: Identifier nickname for evaluation output logging.
            params_override: Optional dictionary of parameter overrides.
        """
        init_time = time.strftime("%Y%m%d%H%M%S")
        logging.info("AGV dispatching environment being initialized...")

        # Load parameters from json
        param_path = pkg_resources.resource_filename('pydispatching', 'agv_data/parameters.json')
        with open(param_path, 'r') as f:
            self._params = json.load(f)

        if params_override:
            self._params.update(params_override)

        # Fleet size resolution
        if num_vehicles is not None:
            self._num_vehicles = int(num_vehicles)
        else:
            self._num_vehicles = int(self._params.get('fleet_size', [9])[0])
        self._V = self._num_vehicles

        # Scenario & request count resolution
        scenario_map = {}
        for sc in self._params.get('request_scenario', []):
            scenario_map.update(sc)

        if request_scenario is not None and request_scenario in scenario_map:
            self._num_requests = int(scenario_map[request_scenario])
        elif num_requests is not None:
            self._num_requests = int(num_requests)
        else:
            self._num_requests = int(scenario_map.get("typical", 900))
        self._R = self._num_requests

        # Objective weight alpha
        if alpha is not None:
            self._alpha = float(np.clip(alpha, 0.0, 1.0))
        else:
            self._alpha = float(self._params.get('alpha', {}).get('value', 0.9))

        # Battery and operational parameters
        self._charging_rate = float(self._params.get('Charging_rate', {}).get('rate', 0.011))
        self._discharge_rate = float(self._params.get('Travel_discharging_rate', {}).get('rate', 0.0055))
        self._critical_battery_level = float(self._params.get('Critical_battery_level', {}).get('level', 20.0))
        self._non_critical_battery_level = float(self._params.get('Non_critical_battery_level', {}).get('level', 80.0))
        self._travel_speed = float(self._params.get('Travel_speed', {}).get('speed', 1.0))
        self._agv_capacity = float(self._params.get('AGV_capacity', {}).get('capacity', 1.0))
        self._request_weight = float(self._params.get('Request_weight', {}).get('weight', 1.0))
        self._time_window_slack = float(self._params.get('Time_window_slack', {}).get('slack', 3600.0))
        self._tardiness_cost_rate = float(self._params.get('Tardiness_cost', {}).get('cost', 1.0))
        self._travel_cost_rate = float(self._params.get('Travel_distance_cost', {}).get('cost', 1.0))

        # Horizon constants
        self._MAX_TIME = float(self._params.get('time_horizon', 86400))
        self._NEVER = self._MAX_TIME + 1.0
        self._MAX_WAIT = float(self._params.get('Maximum_Time_of_Arrival', {}).get('time', 3600.0))

        self._stochastic = stochastic
        self._seed = seed
        self._action_timelimit = action_timelimit
        self._max_interdecision_time = max_interdecision_time if max_interdecision_time is not None else self._NEVER
        self._eval = for_evaluation
        self._nickname = nickname

        self.config = {
            "num_vehicles": self._num_vehicles,
            "num_requests": self._num_requests,
            "alpha": self._alpha,
            "stochastic": stochastic,
            "seed": seed,
            "action_timelimit": action_timelimit,
            "max_interdecision_time": max_interdecision_time,
            "eval": for_evaluation,
            "nickname": nickname,
        }

        # Initialize geometry with stations
        self._geom = AGVGeometry(seed=seed, params=self._params, num_vehicles=self._num_vehicles)

        # Episode and tracking state
        self.curr_episode = -1
        self.curr_step = -1
        self._obs_release_time = None
        self.num_pending_requests = 0

        # Space initialization
        self._make_observation_space()
        self._make_action_space()

        # Rendering setup
        self._prep_rendering()

        # Seeding
        self._initial_seeding()

        # Evaluation file setup
        if self._eval:
            self._eval_out_fname = f"./pydispatching_eval_results_{nickname}_{init_time}.json"
            self._eval_dict = {
                "config": self.config,
                "episodes": []
            }

    def _initial_seeding(self) -> None:
        self._seed_spawner = np.random.SeedSequence(self._seed)
        self._request_sampler = None
        self._vehicle_sampler = None

    def _reseed(self) -> None:
        spawns = self._seed_spawner.spawn(2)
        self._vehicle_sampler = np.random.default_rng(spawns[0])
        self._request_sampler = np.random.default_rng(spawns[1])
        self._geom.reseed()

    @property
    def x_range(self):
        return self._geom.x_range

    @property
    def y_range(self):
        return self._geom.y_range

    @property
    def _NULL_X(self):
        return self.x_range[0]

    @property
    def _NULL_Y(self):
        return self.y_range[0]

    @property
    def num_stations(self):
        return self._geom.num_stations

    @property
    def _D(self):
        return self.num_stations

    @property
    def lots(self):
        return self._geom.lots.loc[:, ["x", "y"]].copy()

    @property
    def curr_state(self):
        return self._make_state()

    @property
    def num_vehicles(self):
        return self._num_vehicles

    @property
    def num_requests(self):
        return self._num_requests

    @property
    def alpha(self):
        return self._alpha

    @property
    def num_stranded_vehicles(self) -> int:
        """Returns number of AGVs currently stranded due to depleted battery (0%)."""
        if hasattr(self, "_vehicles") and "battery" in self._vehicles:
            return int(np.sum(self._vehicles['battery'] <= 1e-4))
        return 0

    def _pending_requests_mask(self) -> pd.Series:
        return (
            self._requests["released"]
            & ~self._requests["rejected"]
            & ~self._requests["completed"]
            & self._requests["vehicle"].isna()
        )

    def _get_pending_requests(self) -> pd.DataFrame:
        return self._requests.loc[self._pending_requests_mask(), :]

    def _get_num_pending_requests(self) -> int:
        return int(self._pending_requests_mask().sum())

    def _make_action_space(self) -> None:
        """Initializes the environment's action space."""
        self.action_space = spaces.Dict({
            # Rejection decision for pending requests (0: keep, 1: reject)
            "req_rejections": InitMultiBinary(self.num_pending_requests),
            # Request assignments: index of vehicle (0..V-1), or V for unassigned
            "req_assgts": spaces.MultiDiscrete(np.full((self.num_pending_requests,), fill_value=self._num_vehicles + 1, dtype=int)),
            # Repositioning: station index (0..num_stations-1), or num_stations for no reposition
            "reposition": spaces.MultiDiscrete(np.full((self._num_vehicles,), fill_value=self.num_stations + 1, dtype=int))
        })

    def _set_action_space(self) -> None:
        """Updates action space when pending request count changes."""
        if self.action_space.spaces["req_rejections"].shape != (self.num_pending_requests,):
            self.action_space.spaces["req_rejections"] = InitMultiBinary(self.num_pending_requests)
            self.action_space.spaces["req_assgts"] = spaces.MultiDiscrete(
                np.full((self.num_pending_requests,), fill_value=self._num_vehicles + 1, dtype=int)
            )

    def _make_observation_space(self) -> gym.Space:
        """Initializes the environment's observation space according to CMDP state space."""
        self.observation_space = spaces.Dict({
            # Current time
            "time": spaces.Box(low=0.0, high=self._MAX_TIME, shape=(1,), dtype=np.float64),

            # Pending requests origin and destination coordinates (M, 2, 2)
            "request_locs": spaces.Box(
                low=np.tile([self.x_range[0], self.y_range[0]], self.num_pending_requests * 2).reshape(self.num_pending_requests, 2, 2) if self.num_pending_requests > 0 else np.empty((0, 2, 2)),
                high=np.tile([self.x_range[1], self.y_range[1]], self.num_pending_requests * 2).reshape(self.num_pending_requests, 2, 2) if self.num_pending_requests > 0 else np.empty((0, 2, 2)),
                dtype=np.float64
            ),

            # Earliest pickup times of pending requests (M,)
            "request_times": spaces.Box(
                low=0.0, high=self._MAX_TIME, shape=(self.num_pending_requests,), dtype=np.float64
            ),

            # Latest delivery times of pending requests (M,)
            "request_deadlines": spaces.Box(
                low=0.0, high=self._MAX_TIME + self._time_window_slack * 2, shape=(self.num_pending_requests,), dtype=np.float64
            ),

            # Loading weights of pending requests (M,)
            "request_weights": spaces.Box(
                low=0.0, high=self._agv_capacity * 2, shape=(self.num_pending_requests,), dtype=np.float64
            ),

            # Vehicle locations (V, 2)
            "v_locs": spaces.Box(
                low=np.tile([self.x_range[0], self.y_range[0]], self._V).reshape(self._V, 2),
                high=np.tile([self.x_range[1], self.y_range[1]], self._V).reshape(self._V, 2),
                dtype=np.float64
            ),

            # Vehicles' job types (V, 3)
            "v_jobs": spaces.MultiDiscrete(np.full((self._V, 3), len(Jobs))),

            # Vehicles' job locations (V, 3, 2, 2)
            "v_job_locs": spaces.Box(
                low=np.tile([self.x_range[0], self.y_range[0]], self._V * 3 * 2).reshape(self._V, 3, 2, 2),
                high=np.tile([self.x_range[1], self.y_range[1]], self._V * 3 * 2).reshape(self._V, 3, 2, 2),
                dtype=np.float64
            ),

            # Vehicle battery levels (V,) in [0, 100]
            "v_battery": spaces.Box(low=0.0, high=100.0, shape=(self._V,), dtype=np.float64),
        })

    def _set_observation_space(self) -> None:
        """Updates observation space dynamically."""
        if self.observation_space.spaces["request_times"].shape != (self.num_pending_requests,):
            self.observation_space.spaces["request_locs"] = spaces.Box(
                low=np.tile([self.x_range[0], self.y_range[0]], self.num_pending_requests * 2).reshape(self.num_pending_requests, 2, 2) if self.num_pending_requests > 0 else np.empty((0, 2, 2)),
                high=np.tile([self.x_range[1], self.y_range[1]], self.num_pending_requests * 2).reshape(self.num_pending_requests, 2, 2) if self.num_pending_requests > 0 else np.empty((0, 2, 2)),
                dtype=np.float64
            )
            self.observation_space.spaces["request_times"] = spaces.Box(
                low=0.0, high=self._MAX_TIME, shape=(self.num_pending_requests,), dtype=np.float64
            )
            self.observation_space.spaces["request_deadlines"] = spaces.Box(
                low=0.0, high=self._MAX_TIME + self._time_window_slack * 2, shape=(self.num_pending_requests,), dtype=np.float64
            )
            self.observation_space.spaces["request_weights"] = spaces.Box(
                low=0.0, high=self._agv_capacity * 2, shape=(self.num_pending_requests,), dtype=np.float64
            )

    def _action_was_slow(self) -> bool:
        if self._obs_release_time is None:
            return False
        return (time.time() - self._obs_release_time) > self._action_timelimit

    def _make_null_jobs(self, mask: Union[np.ndarray, pd.Index, Sequence], jobs: Union[int, List[int]]) -> None:
        """Marks jobs as null for the given vehicles."""
        if isinstance(jobs, int):
            jobs = [jobs]

        xcols = [f"j{n}{od}x" for n in jobs for od in ("o", "d")]
        ycols = [f"j{n}{od}y" for n in jobs for od in ("o", "d")]
        tcols = [f"j{n}{od}t" for n in jobs for od in ("o", "d")]
        mcols = [f"j{n}m" for n in jobs]

        self._vehicles.loc[mask, xcols] = self._NULL_X
        self._vehicles.loc[mask, ycols] = self._NULL_Y
        self._vehicles.loc[mask, tcols] = self._NEVER
        self._vehicles.loc[mask, mcols] = Jobs.NULL

    def _shift_n_jobs(self, mask: np.ndarray, n: int) -> None:
        """Shifts vehicle jobs after completion."""
        orig_dtypes = self._vehicles.dtypes
        if n == 1:
            self._vehicles.loc[mask, ["j1m", "j1ox", "j1oy", "j1ot", "j1dx", "j1dy", "j1dt"]] = (
                self._vehicles.loc[mask, ["j2m", "j2ox", "j2oy", "j2ot", "j2dx", "j2dy", "j2dt"]].to_numpy()
            )
            self._vehicles.loc[mask, ["j2m", "j2ox", "j2oy", "j2ot", "j2dx", "j2dy", "j2dt"]] = (
                self._vehicles.loc[mask, ["j3m", "j3ox", "j3oy", "j3ot", "j3dx", "j3dy", "j3dt"]].to_numpy()
            )
            self._make_null_jobs(mask, 3)
        elif n == 2:
            self._vehicles.loc[mask, ["j1m", "j1ox", "j1oy", "j1ot", "j1dx", "j1dy", "j1dt"]] = (
                self._vehicles.loc[mask, ["j3m", "j3ox", "j3oy", "j3ot", "j3dx", "j3dy", "j3dt"]].to_numpy()
            )
            self._make_null_jobs(mask, [2, 3])
        elif n == 3:
            self._make_null_jobs(mask, [1, 2, 3])
        else:
            raise ValueError(f"Invalid value for n: {n}")
        self._vehicles = self._vehicles.astype(orig_dtypes)

    def _check_valid_action(self, action: Dict[str, np.ndarray]) -> None:
        """Validates action structure and ensures no vehicle is assigned multiple requests."""
        if not self.action_space.contains(action):
            raise ValueError("Invalid action provided: Not contained in action space.")

        # Ensure vehicle is not assigned to more than one request in the same epoch
        req_assgts = action["req_assgts"]
        actual_assgts = req_assgts[req_assgts != self._V]
        if len(np.unique(actual_assgts)) != len(actual_assgts):
            raise ValueError("Invalid action: An AGV was assigned to more than one request simultaneously.")

    def _check_assignment_feasibility(self, vs: pd.DataFrame, reqs: pd.DataFrame) -> np.ndarray:
        """Checks instantaneous constraints (battery threshold, capacity) and job queue."""
        if len(vs) == 0:
            return np.full((0,), fill_value=True, dtype=bool)

        # 1. Critical Battery Level constraint (Instantaneous constraint masking)
        # If battery < critical_battery_level, vehicle CANNOT accept new transport tasks
        battery_ok = vs["battery"].to_numpy() >= self._critical_battery_level

        # 2. Capacity constraint (Instantaneous constraint)
        capacity_ok = reqs["weight"].to_numpy() <= vs["capacity"].to_numpy()

        # 3. Job Feasibility: Vehicle cannot have a second non-preemptible job already lined up
        job_feasible = np.isin(vs["j2m"].to_numpy(), (Jobs.IDLE, Jobs.REPOSITION, Jobs.CHARGING, Jobs.NULL))

        # 4. Total Battery Sufficiency for task:
        dist_to_origin = self._geom.dist(
            o=vs[["avail_x", "avail_y"]].to_numpy(),
            d=reqs[["ox", "oy"]].to_numpy(),
            pairwise=False
        )
        dist_origin_dest = self._geom.dist(
            o=reqs[["ox", "oy"]].to_numpy(),
            d=reqs[["dx", "dy"]].to_numpy(),
            pairwise=False
        )
        total_dist = dist_to_origin + dist_origin_dest
        total_time = total_dist / self._travel_speed
        battery_consumed = total_time * self._discharge_rate
        predicted_battery = vs["battery"].to_numpy() - battery_consumed
        battery_sufficient = predicted_battery >= self._critical_battery_level

        return battery_ok & capacity_ok & job_feasible & battery_sufficient

    def _get_server_job_col_updates(self, req_idxs: pd.Index, v_idxs: np.ndarray) -> np.ndarray:
        """Computes job table updates for vehicles serving assigned requests."""
        first_job_oloc = self._vehicles.loc[v_idxs, ["avail_x", "avail_y"]].to_numpy()
        first_job_dloc = self._requests.loc[req_idxs, ["ox", "oy"]].to_numpy()
        first_job_ot = self._vehicles.loc[v_idxs, "avail_t"].to_numpy()
        first_job_dt = first_job_ot + self._geom.travel_time(
            o=first_job_oloc,
            d=first_job_dloc,
            pairwise=False,
        )

        second_job_oloc = first_job_dloc
        second_job_dloc = self._requests.loc[req_idxs, ["dx", "dy"]].to_numpy()
        second_job_ot = first_job_dt
        second_job_dt = second_job_ot + self._requests.loc[req_idxs, "pt"].to_numpy()

        avail_loc = second_job_dloc
        avail_t = second_job_dt
        epoch_t = avail_t

        first_job_hstack = np.hstack([
            first_job_oloc,
            first_job_ot.reshape(-1, 1),
            first_job_dloc,
            first_job_dt.reshape(-1, 1)
        ])
        second_job_hstack = np.hstack([
            second_job_oloc,
            second_job_ot.reshape(-1, 1),
            second_job_dloc,
            second_job_dt.reshape(-1, 1)
        ])
        full_hstack = np.hstack([
            first_job_hstack,
            second_job_hstack,
            avail_loc,
            avail_t.reshape(-1, 1),
            epoch_t.reshape(-1, 1)
        ])
        return full_hstack

    def _update_servers_job_cols(self, req_idxs: pd.Index, v_idxs: np.ndarray) -> None:
        """Updates vehicle scheduling columns for service assignments."""
        if len(req_idxs) == 0:
            return

        job_col_updates = self._get_server_job_col_updates(req_idxs, v_idxs)
        server_busy_mask = (self._vehicles.loc[v_idxs, 'avail_t'] != self.time).to_numpy()

        busy_servers = v_idxs[server_busy_mask]
        now_servers = v_idxs[~server_busy_mask]

        if len(busy_servers) > 0:
            self._vehicles.loc[busy_servers, ["j2m", "j3m"]] = Jobs.SETUP, Jobs.PROCESS
            update_cols = (
                [f"j{j_idx}{od}{xyt}" for j_idx in (2, 3) for od in ("o", "d") for xyt in ("x", "y", "t")]
                + ["avail_x", "avail_y", "avail_t", "epoch_t"]
            )
            self._vehicles.loc[busy_servers, update_cols] = job_col_updates[server_busy_mask]

        if len(now_servers) > 0:
            self._vehicles.loc[now_servers, ["j1m", "j2m"]] = Jobs.SETUP, Jobs.PROCESS
            update_cols = (
                [f"j{j_idx}{od}{xyt}" for j_idx in (1, 2) for od in ("o", "d") for xyt in ("x", "y", "t")]
                + ["avail_x", "avail_y", "avail_t", "epoch_t"]
            )
            self._vehicles.loc[now_servers, update_cols] = job_col_updates[~server_busy_mask]
            self._make_null_jobs(mask=now_servers, jobs=3)

    def _update_repos_job_cols(self, repos_v_idxs: pd.Index, target_station_ids: np.ndarray) -> None:
        """Updates vehicle scheduling columns for repositioning / charging moves."""
        if len(repos_v_idxs) == 0:
            return

        charging_locs = self._geom.charging_stations[['x', 'y']].to_numpy()

        for v_idx, st_id in zip(repos_v_idxs, target_station_ids):
            curr_loc = self._vehicles.loc[v_idx, ["x", "y"]].to_numpy(dtype=float)
            curr_bat = self._vehicles.at[v_idx, "battery"]
            target_lot = self.lots.loc[st_id, ["x", "y"]].to_numpy(dtype=float)
            dist = np.abs(curr_loc - target_lot).sum()
            dest_is_charging = bool(np.any(np.all(np.abs(charging_locs - target_lot) < 1e-3, axis=1)))

            # If vehicle has no battery (stranded) and is not already at station, it cannot travel
            if curr_bat <= 1e-4 and dist > 1e-3:
                continue

            if dist < 1e-3:
                # Vehicle is already at the target station - no travel needed
                new_j1m = Jobs.CHARGING if dest_is_charging else Jobs.IDLE
                self._vehicles.at[v_idx, "j1m"] = new_j1m
                self._vehicles.at[v_idx, "j2m"] = Jobs.NULL
                self._vehicles.at[v_idx, "j3m"] = Jobs.NULL
                self._vehicles.loc[v_idx, ["j1ox", "j1oy", "j1dx", "j1dy"]] = [curr_loc[0], curr_loc[1], curr_loc[0], curr_loc[1]]
                self._vehicles.at[v_idx, "j1ot"] = self.time
                self._vehicles.at[v_idx, "j1dt"] = self._NEVER
                self._vehicles.loc[v_idx, ["avail_x", "avail_y"]] = [curr_loc[0], curr_loc[1]]
                self._vehicles.at[v_idx, "avail_t"] = self.time
                self._vehicles.at[v_idx, "epoch_t"] = self._NEVER
            else:
                travel_time = float(dist / self._travel_speed)
                arr_time = self.time + travel_time

                self._vehicles.at[v_idx, "j1m"] = Jobs.REPOSITION
                self._vehicles.at[v_idx, "j2m"] = Jobs.CHARGING if dest_is_charging else Jobs.IDLE
                self._vehicles.at[v_idx, "j3m"] = Jobs.NULL

                self._vehicles.loc[v_idx, ["j1ox", "j1oy"]] = [curr_loc[0], curr_loc[1]]
                self._vehicles.loc[v_idx, ["j1dx", "j1dy"]] = [target_lot[0], target_lot[1]]
                self._vehicles.at[v_idx, "j1ot"] = self.time
                self._vehicles.at[v_idx, "j1dt"] = arr_time

                self._vehicles.loc[v_idx, ["j2ox", "j2oy", "j2dx", "j2dy"]] = [target_lot[0], target_lot[1], target_lot[0], target_lot[1]]
                self._vehicles.at[v_idx, "j2ot"] = arr_time
                self._vehicles.at[v_idx, "j2dt"] = self._NEVER

                self._vehicles.loc[v_idx, ["avail_x", "avail_y"]] = [target_lot[0], target_lot[1]]
                self._vehicles.at[v_idx, "avail_t"] = arr_time
                self._vehicles.at[v_idx, "epoch_t"] = arr_time

        self._make_null_jobs(mask=repos_v_idxs, jobs=3)

    def _get_jobs_dists(self) -> np.ndarray:
        """Returns (V, 3) distances for jobs currently assigned to vehicles."""
        return np.hstack([
            self._geom.dist(
                o=self._vehicles[["j1ox", "j1oy"]].to_numpy(),
                d=self._vehicles[["j1dx", "j1dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1, 1),
            self._geom.dist(
                o=self._vehicles[["j2ox", "j2oy"]].to_numpy(),
                d=self._vehicles[["j2dx", "j2dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1, 1),
            self._geom.dist(
                o=self._vehicles[["j3ox", "j3oy"]].to_numpy(),
                d=self._vehicles[["j3dx", "j3dy"]].to_numpy(),
                pairwise=False
            ).reshape(-1, 1)
        ])

    def step(self, action: Dict[str, np.ndarray]):
        """Advance the environment by taking action.
        
        Evaluates tardiness cost and travel distance cost according to CMDP objective:
        min alpha * sum(c_r * tau_rv) + (1 - alpha) * sum(c_v * delta_rv)
        """
        if self._action_was_slow():
            logging.warning("Took too long to provide action.")
            return self._make_state(), -np.inf, True, {"error": "action_timeout"}

        self._check_valid_action(action)

        reqs = self._get_pending_requests()
        reqs_idx = reqs.index

        #
        # Phase I: Transition Updates from Action
        #
        # Process request assignments with feasibility checks
        req_assgts = np.where(action['req_rejections'].astype(bool), self._V, action['req_assgts'])

        assgd_reqs = reqs.loc[req_assgts != self._V, :]
        serving_vs = self._vehicles.loc[req_assgts[req_assgts != self._V], :]
        infeasible = ~self._check_assignment_feasibility(serving_vs, assgd_reqs)

        if np.any(infeasible):
            bad_vs = serving_vs.index[infeasible]
            req_assgts = np.where(np.isin(req_assgts, bad_vs), self._V, req_assgts)

        rejected_reqs = reqs_idx[action["req_rejections"].astype(bool)]
        assgd_reqs_mask = req_assgts != self._V
        assgd_reqs_idxs = reqs_idx[assgd_reqs_mask]
        serving_idxs = req_assgts[assgd_reqs_mask]

        # Enqueue assigned requests for vehicles
        for r_id, v_id in zip(assgd_reqs_idxs, serving_idxs):
            self._v_req_queue[v_id].append(r_id)

        self._update_servers_job_cols(assgd_reqs_idxs, serving_idxs)
        self._requests.loc[rejected_reqs, "rejected"] = True
        self._requests.loc[assgd_reqs_idxs, "vehicle"] = serving_idxs

        # Process repositioning
        reposs = np.array(action["reposition"], copy=True)
        has_repos_assgt = reposs != self._D

        # Do not allow repositioning for vehicles busy serving
        bad_repos_assgt = has_repos_assgt & (self._vehicles['avail_t'] != self.time).to_numpy()
        if np.any(bad_repos_assgt):
            reposs[bad_repos_assgt] = self._D
            has_repos_assgt = reposs != self._D

        repos_v_idxs = self._vehicles.index[has_repos_assgt]
        self._update_repos_job_cols(repos_v_idxs, reposs[has_repos_assgt])

        #
        # Phase I.III: Determine Next Decision Epoch Time
        #
        future_ev_mask = self._vehicles["epoch_t"] > self.time
        next_ev_epoch_time = float(self._vehicles.loc[future_ev_mask, "epoch_t"].min()) if np.any(future_ev_mask) else self._NEVER

        next_req_epoch_time = float(self._requests.at[self._next_request_idx, "time"]) if self._next_request_idx < len(self._requests) else self._NEVER

        # Check charging completion times for vehicles currently charging
        charging_mask = self._vehicles["j1m"] == Jobs.CHARGING
        if np.any(charging_mask):
            charging_bats = self._vehicles.loc[charging_mask, "battery"].to_numpy()
            needed_charge = np.maximum(0.0, 100.0 - charging_bats)
            chg_durations = needed_charge / self._charging_rate
            future_chg_durations = chg_durations[chg_durations > 0]
            next_chg_epoch_time = float(self.time + future_chg_durations.min()) if len(future_chg_durations) > 0 else self._NEVER
        else:
            next_chg_epoch_time = self._NEVER

        # Check pending request deadlines
        pending_reqs = self._get_pending_requests()
        if not pending_reqs.empty:
            future_deadlines = pending_reqs.loc[pending_reqs["deadline"] > self.time, "deadline"]
            next_deadline_time = float(future_deadlines.min()) if not future_deadlines.empty else self._NEVER
        else:
            next_deadline_time = self._NEVER

        # Periodic interdecision step when vehicles are actively moving
        has_traveling_evs = np.any(self._vehicles["j1m"].isin([Jobs.REPOSITION, Jobs.SETUP, Jobs.PROCESS]))
        if has_traveling_evs:
            next_forced_epoch_time = min(self._MAX_TIME, self.time + self._max_interdecision_time)
        else:
            next_forced_epoch_time = self._NEVER

        candidate_times = [
            t for t in (next_ev_epoch_time, next_req_epoch_time, next_chg_epoch_time, next_deadline_time, next_forced_epoch_time)
            if t > self.time
        ]
        if candidate_times:
            next_epoch_time = min(candidate_times)
        else:
            next_epoch_time = self._MAX_TIME

        is_new_request = (next_epoch_time == next_req_epoch_time) and (next_epoch_time <= self._MAX_TIME)

        delta_t = next_epoch_time - self.time

        #
        # Phase II: Simulation Advancement over delta_t
        #
        job_otimes = self._vehicles[["j1ot", "j2ot", "j3ot"]].to_numpy()
        job_dtimes = self._vehicles[["j1dt", "j2dt", "j3dt"]].to_numpy()
        job_durations = job_dtimes - job_otimes
        job_durations_safe = np.where(job_durations == 0, np.inf, job_durations)

        job_pct_remaining_now = np.clip(job_dtimes - self.time, 0, job_durations) / job_durations_safe
        job_pct_remaining_next = np.clip(job_dtimes - next_epoch_time, 0, job_durations) / job_durations_safe
        job_pct_completed = job_pct_remaining_now - job_pct_remaining_next
        job_pct_completed = np.nan_to_num(job_pct_completed, nan=0.0)

        job_dists = self._get_jobs_dists()
        dists_traveled = job_pct_completed * job_dists
        step_travel_distance = float(dists_traveled.sum())

        # Check completed jobs & requests
        num_to_remove = (next_epoch_time >= job_dtimes).sum(axis=1)

        # Identify active requests at step start (released, uncompleted, non-sentinel)
        active_mask = self._requests["released"] & ~self._requests["completed"] & (self._requests["time"] != self._NEVER)

        # Step tardiness cost and delivery tracking (Option B: Continuous Accumulation)
        step_tardiness_cost = 0.0
        newly_completed_req_count = 0

        # 1. Process vehicles completing a PROCESS job (delivery) in this transition
        for v_idx in range(self._V):
            j1_type = self._vehicles.at[v_idx, "j1m"]
            j1_dt = self._vehicles.at[v_idx, "j1dt"]
            if j1_type == Jobs.PROCESS and j1_dt <= next_epoch_time:
                if self._v_req_queue[v_idx]:
                    req_id = self._v_req_queue[v_idx].pop(0)
                    self._requests.at[req_id, "completed"] = True
                    self._requests.at[req_id, "delivery_time"] = j1_dt
                    deadline = float(self._requests.at[req_id, "deadline"])
                    cost_rate = float(self._requests.at[req_id, "cost_rate"])

                    # Tardiness accrued during this step up to the exact delivery timestamp j1_dt
                    delta_tardiness = max(0.0, j1_dt - max(self.time, deadline))
                    self._requests.at[req_id, "tardiness"] += delta_tardiness
                    step_tardiness_cost += cost_rate * delta_tardiness
                    newly_completed_req_count += 1

        self.num_completed_requests += newly_completed_req_count

        # 2. Continuous tardiness accumulation for all other active requests over [self.time, next_epoch_time]
        uncompleted_mask = active_mask & ~self._requests["completed"]
        if uncompleted_mask.any():
            uncompleted_reqs = self._requests[uncompleted_mask]
            deadlines = uncompleted_reqs["deadline"].to_numpy(dtype=float)
            cost_rates = uncompleted_reqs["cost_rate"].to_numpy(dtype=float)
            delta_tau = np.maximum(0.0, next_epoch_time - np.maximum(self.time, deadlines))
            pos_mask = delta_tau > 0.0
            if np.any(pos_mask):
                pos_indices = uncompleted_reqs.index[pos_mask]
                pos_delta_tau = delta_tau[pos_mask]
                pos_cost_rates = cost_rates[pos_mask]
                self._requests.loc[pos_indices, "tardiness"] += pos_delta_tau
                step_tardiness_cost += float(np.sum(pos_cost_rates * pos_delta_tau))

        # Shift jobs
        for n in (1, 2, 3):
            if np.any(num_to_remove == n):
                self._shift_n_jobs(mask=(num_to_remove == n), n=n)

        # Update epoch_t for vehicles based on their active schedule
        for v in range(self._V):
            j1 = self._vehicles.at[v, "j1m"]
            j2 = self._vehicles.at[v, "j2m"]
            if j1 in (Jobs.IDLE, Jobs.CHARGING, Jobs.NULL) and j2 == Jobs.NULL:
                self._vehicles.at[v, "epoch_t"] = self._NEVER
            elif j2 in (Jobs.SETUP, Jobs.PROCESS):
                self._vehicles.at[v, "epoch_t"] = self._vehicles.at[v, "j2dt"]
            elif j1 in (Jobs.SETUP, Jobs.PROCESS, Jobs.REPOSITION):
                self._vehicles.at[v, "epoch_t"] = self._vehicles.at[v, "j1dt"]
            else:
                self._vehicles.at[v, "epoch_t"] = self._NEVER

        # Update vehicle positions
        finishing_vs = self._vehicles["j1m"] == Jobs.NULL
        if np.any(finishing_vs):
            self._vehicles.loc[finishing_vs, ["x", "y"]] = self._vehicles.loc[finishing_vs, ["avail_x", "avail_y"]].to_numpy()

        progress_vs = ~finishing_vs
        if np.any(progress_vs):
            denom = (self._vehicles.loc[progress_vs, "j1dt"] - self._vehicles.loc[progress_vs, "j1ot"]).to_numpy()
            denom_safe = np.where(denom == 0, 1.0, denom)
            rel_elapsed = np.clip((next_epoch_time - self._vehicles.loc[progress_vs, "j1ot"]).to_numpy() / denom_safe, 0.0, 1.0)
            orig_coords = self._vehicles.loc[progress_vs, ["j1ox", "j1oy"]].to_numpy()
            dest_coords = self._vehicles.loc[progress_vs, ["j1dx", "j1dy"]].to_numpy()
            self._vehicles.loc[progress_vs, ["x", "y"]] = orig_coords + rel_elapsed[:, None] * (dest_coords - orig_coords)

        # Available time/loc update for non-busy (stationary) vehicles
        not_busy = self._vehicles["j1m"].isin((Jobs.IDLE, Jobs.CHARGING))
        self._vehicles.loc[not_busy, ["avail_x", "avail_y"]] = self._vehicles.loc[not_busy, ["x", "y"]].to_numpy()
        self._vehicles.loc[not_busy, "avail_t"] = next_epoch_time

        #
        # Phase II.III: Battery Management & Dynamics
        #
        if delta_t > 0:
            is_traveling = self._vehicles['j1m'].isin([Jobs.REPOSITION, Jobs.SETUP, Jobs.PROCESS])
            is_idle_or_charging = self._vehicles['j1m'].isin([Jobs.IDLE, Jobs.CHARGING])

            # Discharge during transit
            self._vehicles.loc[is_traveling, 'battery'] -= self._discharge_rate * delta_t

            # Recharge when at charging stations
            charging_stations = self._geom.charging_stations[['x', 'y']].to_numpy()
            idle_charging_indices = self._vehicles[is_idle_or_charging].index
            if len(idle_charging_indices) > 0 and len(charging_stations) > 0:
                v_locs = self._vehicles.loc[idle_charging_indices, ['x', 'y']].to_numpy()
                dist_matrix = distance.cdist(v_locs, charging_stations, 'cityblock')
                at_charger = dist_matrix.min(axis=1) < 1e-3
                charging_v_idxs = idle_charging_indices[at_charger]
                if len(charging_v_idxs) > 0:
                    self._vehicles.loc[charging_v_idxs, 'battery'] += self._charging_rate * delta_t
                    # Update status to CHARGING if idle at charger
                    idle_mask = self._vehicles.loc[charging_v_idxs, 'j1m'] == Jobs.IDLE
                    if np.any(idle_mask):
                        self._vehicles.loc[charging_v_idxs[idle_mask], 'j1m'] = Jobs.CHARGING

            self._vehicles['battery'] = self._vehicles['battery'].clip(0.0, 100.0)

        #
        # Phase III: Reward & Cost Formulation (CMDP Section 4.2)
        #
        step_travel_cost = self._travel_cost_rate * step_travel_distance
        step_cost = self._alpha * step_tardiness_cost + (1.0 - self._alpha) * step_travel_cost
        step_reward = -step_cost

        self.total_travel_distance += step_travel_distance
        self.total_travel_cost += step_travel_cost
        self.total_tardiness_cost += step_tardiness_cost
        self.total_cost += step_cost
        self.rewards += step_reward

        #
        # Phase IV: Advance Time & State
        #
        self.time = next_epoch_time

        if is_new_request and self._next_request_idx < len(self._requests):
            self._requests.at[self._next_request_idx, "released"] = True
            self._next_request_idx += 1

        self.curr_step += 1
        self.num_pending_requests = self._get_num_pending_requests()
        self._set_action_space()
        self._set_observation_space()

        obs = self._make_state()

        # Check robust termination conditions
        is_time_exhausted = self.time >= self._MAX_TIME
        all_released = self._next_request_idx >= len(self._requests) - 1
        has_moving_vehicles = bool(np.any(self._vehicles['j1m'].isin([Jobs.REPOSITION, Jobs.SETUP, Jobs.PROCESS])))
        has_queued_jobs = bool(np.any(self._vehicles['j2m'].isin([Jobs.SETUP, Jobs.PROCESS])))

        # 1. Complete task finish: All requests released, queue empty, and all AGVs done
        all_requests_done = all_released and (self.num_pending_requests == 0) and (not has_moving_vehicles) and (not has_queued_jobs)

        # 2. Stagnation timeout: All requests released, fleet completely idle, and time has passed last deadline + 30 min grace period
        actual_reqs = self._requests[self._requests['time'] != self._NEVER]
        last_deadline = float(actual_reqs['deadline'].max()) if len(actual_reqs) > 0 else self._MAX_TIME
        stagnant_timeout = all_released and (not has_moving_vehicles) and (not has_queued_jobs) and (self.time >= last_deadline + 1800.0)

        terminal = bool(is_time_exhausted or all_requests_done or stagnant_timeout)

        info = {
            "step_travel_cost": step_travel_cost,
            "step_tardiness_cost": step_tardiness_cost,
            "step_cost": step_cost,
            "total_travel_distance": self.total_travel_distance,
            "total_travel_cost": self.total_travel_cost,
            "total_tardiness_cost": self.total_tardiness_cost,
            "total_cost": self.total_cost,
            "num_completed_requests": self.num_completed_requests,
            "num_pending_requests": self.num_pending_requests,
            "mean_battery": float(self._vehicles['battery'].mean()),
            "min_battery": float(self._vehicles['battery'].min()),
            "num_stranded_vehicles": self.num_stranded_vehicles,
            "alpha": self._alpha,
        }

        if self._eval and terminal:
            total_reqs = max(1, len(self._requests) - 1)
            comp_pct = min(100.0, (self.num_completed_requests / total_reqs) * 100.0)
            stranded_pct = (self.num_stranded_vehicles / max(1, self._V)) * 100.0

            self.episode_dict["final_reward"] = self.rewards
            self.episode_dict["final_cost"] = self.total_cost
            self.episode_dict["travel_distance"] = self.total_travel_distance
            self.episode_dict["tardiness_cost"] = self.total_tardiness_cost
            self.episode_dict["completed_requests"] = self.num_completed_requests
            self.episode_dict["total_requests"] = total_reqs
            self.episode_dict["completion_rate_pct"] = round(comp_pct, 2)
            self.episode_dict["stranded_vehicles"] = self.num_stranded_vehicles
            self.episode_dict["stranded_rate_pct"] = round(stranded_pct, 2)
            self._eval_dict["episodes"].append(self.episode_dict)
            self._write_eval_dict()

        self._obs_release_time = time.time()
        return obs, step_reward, terminal, info

    def _write_eval_dict(self):
        with open(self._eval_out_fname, "w") as f:
            json.dump(self._eval_dict, f, cls=NpEncoder)

    def reset(self):
        """Sets the environment to an initial state and returns observation."""
        self.curr_episode += 1
        self.curr_step = 0
        self.time = 0.0
        self.rewards = 0.0
        self.total_travel_distance = 0.0
        self.total_travel_cost = 0.0
        self.total_tardiness_cost = 0.0
        self.total_cost = 0.0
        self.num_completed_requests = 0

        self._reseed()
        self._initialize_vehicles()
        self._generate_requests()

        self.num_pending_requests = 0
        self._set_observation_space()
        self._set_action_space()
        self._next_request_idx = 0

        obs = self._make_state()
        self._obs_release_time = None

        if self._eval:
            self.episode_dict = {
                "episode": self.curr_episode,
                "assignments": [],
                "final_reward": -np.inf,
            }

        return obs

    def _make_state(self) -> Dict[str, np.ndarray]:
        """Creates the agent observation matching CMDP state space."""
        curr_requests = self._get_pending_requests()
        n_pending = len(curr_requests)

        if n_pending > 0:
            req_locs = curr_requests[["ox", "oy", "dx", "dy"]].to_numpy().reshape(n_pending, 2, 2).astype(np.float64)
            req_times = curr_requests["time"].to_numpy().astype(np.float64)
            req_deadlines = curr_requests["deadline"].to_numpy().astype(np.float64)
            req_weights = curr_requests["weight"].to_numpy().astype(np.float64)
        else:
            req_locs = np.empty((0, 2, 2), dtype=np.float64)
            req_times = np.empty((0,), dtype=np.float64)
            req_deadlines = np.empty((0,), dtype=np.float64)
            req_weights = np.empty((0,), dtype=np.float64)

        v_locs = self._vehicles[["x", "y"]].to_numpy().astype(np.float64)
        v_jobs = self._vehicles[["j1m", "j2m", "j3m"]].to_numpy().astype(int)
        v_job_locs = self._vehicles[[
            "j1ox", "j1oy", "j1dx", "j1dy",
            "j2ox", "j2oy", "j2dx", "j2dy",
            "j3ox", "j3oy", "j3dx", "j3dy"
        ]].to_numpy().reshape(self._V, 3, 2, 2).astype(np.float64)
        v_battery = self._vehicles["battery"].to_numpy().astype(np.float64)

        return {
            "time": np.array([self.time], dtype=np.float64),
            "request_locs": req_locs,
            "request_times": req_times,
            "request_deadlines": req_deadlines,
            "request_weights": req_weights,
            "v_locs": v_locs,
            "v_jobs": v_jobs,
            "v_job_locs": v_job_locs,
            "v_battery": v_battery,
        }

    def _get_assgd_req_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns origin and destination coordinates of requests assigned to vehicles."""
        assgd_reqs_list = []
        for j_idx in (1, 2, 3):
            mask = self._vehicles[f"j{j_idx}m"] == Jobs.PROCESS
            if mask.any():
                assgd_reqs_list.append(
                    self._vehicles.loc[
                        mask,
                        [f"j{j_idx}ox", f"j{j_idx}oy", f"j{j_idx}dx", f"j{j_idx}dy"]
                    ].to_numpy()
                )
        if not assgd_reqs_list:
            return np.empty((0, 2)), np.empty((0, 2))
        assgd_reqs = np.vstack(assgd_reqs_list)
        return assgd_reqs[:, :2], assgd_reqs[:, 2:]

    def _get_pending_req_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns origin and destination coordinates of pending unassigned requests."""
        pending_reqs = self._get_pending_requests()
        if pending_reqs.empty:
            return np.empty((0, 2)), np.empty((0, 2))
        return pending_reqs[["ox", "oy"]].to_numpy(), pending_reqs[["dx", "dy"]].to_numpy()

    def _prep_rendering(self) -> None:
        """Pre-renders layout stations matching Figure 1 and Figure 2."""
        self._img_width_px = 2080
        self._img_height_px = 1260

        self._min_x = self.x_range[0]
        self._span_x = self.x_range[1] - self.x_range[0]
        self._min_y = self.y_range[0]
        self._span_y = self.y_range[1] - self.y_range[0]

        self._scale_x = self._img_width_px / self._span_x if self._span_x > 1e-6 else 1.0
        self._scale_y = self._img_height_px / self._span_y if self._span_y > 1e-6 else 1.0

        def _draw_station(station_data: pd.Series, drw) -> None:
            station_size_px = 32
            half_size = station_size_px / 2

            center_x = (station_data['x'] - self._min_x) * self._scale_x
            center_y = (station_data['y'] - self._min_y) * self._scale_y

            x0, y0 = center_x - half_size, center_y - half_size
            x1, y1 = center_x + half_size, center_y + half_size

            station_type = station_data['type']
            color_map = {
                'Pickup and Delivery': (173, 216, 230),  # Light blue
                'Charging station': (144, 238, 144),     # Light green
                'Parking station': (255, 165, 0)         # Orange
            }
            fill_color = color_map.get(station_type, (200, 200, 200))
            drw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=(0, 0, 0), width=2)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()

            if station_type == 'Charging station':
                drw.text((center_x - 5, center_y - 8), font=font, text='C', fill=(0, 0, 0))
            elif station_type == 'Parking station':
                drw.text((center_x - 5, center_y - 8), font=font, text='P', fill=(0, 0, 0))

        img = Image.new('RGB', (self._img_width_px, self._img_height_px), color=(255, 255, 255))
        drw = ImageDraw.Draw(img, 'RGBA')

        if not self._geom.stations.empty:
            self._geom.stations.apply(_draw_station, args=(drw,), axis=1)

        del drw
        self._base_render = np.array(img)

    def _draw_reqs(self, drw, req_os: np.ndarray, req_ds: np.ndarray, color: Tuple[int, int, int], radius: int = 6):
        """Draws request lines with circle at origin and triangle at destination."""
        if len(req_os) == 0:
            return

        oxs = (req_os[:, 0] - self._min_x) * self._scale_x
        oys = (req_os[:, 1] - self._min_y) * self._scale_y
        dxs = (req_ds[:, 0] - self._min_x) * self._scale_x
        dys = (req_ds[:, 1] - self._min_y) * self._scale_y

        for ox, oy, dx, dy in zip(oxs, oys, dxs, dys):
            if np.isfinite([ox, oy, dx, dy]).all():
                # Connecting line
                drw.line([(ox, oy), (dx, dy)], fill=color, width=2)
                # Origin circle
                drw.ellipse([ox - radius, oy - radius, ox + radius, oy + radius], fill=color, outline=(0, 0, 0), width=1)
                # Destination triangle
                p1 = (dx, dy - radius * 1.5)
                p2 = (dx - radius * 1.2, dy + radius * 0.8)
                p3 = (dx + radius * 1.2, dy + radius * 0.8)
                drw.polygon([p1, p2, p3], fill=color, outline=(0, 0, 0))

    def _draw_vehicles(self, drw, size_px: int = 34, fill=(255, 215, 0), outline=(0, 0, 0)):
        """Draws AGV circles with visual indication for battery state (normal, critical, depleted)."""
        half_size = size_px / 2
        xs = (self._vehicles["x"] - self._min_x) * self._scale_x
        ys = (self._vehicles["y"] - self._min_y) * self._scale_y
        batteries = self._vehicles["battery"].to_numpy()

        for v_idx, (x, y, bat) in enumerate(zip(xs, ys, batteries)):
            if np.isfinite([x, y]).all():
                if bat <= 0.0:
                    v_fill = (128, 128, 128)  # Gray for depleted
                    v_outline = (220, 20, 60)  # Crimson border
                elif bat <= self._critical_battery_level:
                    v_fill = (255, 140, 0)  # Orange for critical (<20%)
                    v_outline = (220, 20, 60)
                else:
                    v_fill = fill  # Standard yellow
                    v_outline = outline

                drw.ellipse([x - half_size, y - half_size, x + half_size, y + half_size], fill=v_fill, outline=v_outline, width=2)

    def render(self, mode='rgb_array', close=False):
        """Renders environment frame."""
        img = Image.fromarray(self._base_render)
        drw = ImageDraw.Draw(img, 'RGBA')

        # Assigned requests (light red / coral)
        req_os, req_ds = self._get_assgd_req_locs()
        if len(req_os) > 0:
            self._draw_reqs(drw, req_os, req_ds, color=(240, 128, 128))

        # Unassigned pending requests (dark red)
        req_os, req_ds = self._get_pending_req_locs()
        if len(req_os) > 0:
            self._draw_reqs(drw, req_os, req_ds, color=(178, 34, 34))

        # Vehicles (yellow / orange / gray)
        self._draw_vehicles(drw)

        del drw
        return np.array(img)

    def draw_label(self, frame: np.ndarray, episode_num: int, reward: float, info: Optional[Dict[str, Any]] = None) -> Image.Image:
        """Draws statistics overlay matching manuscript Figure 1 with Completed and Depleted AGV counts."""
        im = Image.fromarray(frame)
        drawer = ImageDraw.Draw(im, 'RGBA')
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        total_reqs = len(self._requests) - 1 if hasattr(self, '_requests') and len(self._requests) > 1 else self._num_requests
        v_bat = self._vehicles["battery"].to_numpy() if hasattr(self, "_vehicles") else np.array([])
        depleted_count = int(np.sum(v_bat <= 0.0))

        if info:
            label_text = (
                f"Episode: {episode_num} | Dist: {self.total_travel_distance:.1f} | "
                f"Tardiness Cost: {self.total_tardiness_cost:.1f} | Total Cost: {self.total_cost:.1f} | "
                f"Completed: {self.num_completed_requests}/{total_reqs} | "
                f"Depleted AGVs: {depleted_count}/{self._V}"
            )
        else:
            label_text = (
                f"Episode: {episode_num} | Dist: {self.total_travel_distance:.1f} | "
                f"Completed: {self.num_completed_requests}/{total_reqs} | "
                f"Depleted AGVs: {depleted_count}/{self._V}"
            )

        # Compute text dimensions for centered placement and badge padding
        bbox = drawer.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Shift up to pos_y = 0.895 (away from bottom parking and charging stations)
        pos_x = int((im.size[0] - text_w) / 2)
        pos_y = int(im.size[1] * 0.895)

        # Draw semi-transparent pill container background for clear legibility
        pad_x, pad_y = 16, 8
        badge_box = [pos_x - pad_x, pos_y - pad_y, pos_x + text_w + pad_x, pos_y + text_h + pad_y]
        drawer.rounded_rectangle(badge_box, radius=8, fill=(255, 255, 255, 220), outline=(180, 180, 180, 255), width=1)
        drawer.text((pos_x, pos_y), font=font, text=label_text, fill=(20, 20, 20))
        return im

    def _initialize_vehicles(self) -> None:
        """Initializes AGV fleet positions, battery (100%), capacity, and jobs."""
        self._v_req_queue = {v: [] for v in range(self._num_vehicles)}
        init_locs = self._geom.vehicle_locations[:self._num_vehicles]
        vehicles = pd.DataFrame(init_locs, columns=['x', 'y'], index=pd.RangeIndex(start=0, stop=self._num_vehicles))

        vehicles["j1m"] = Jobs.IDLE
        vehicles[["j1ox", "j1oy"]] = vehicles[["x", "y"]]
        vehicles[["j1dx", "j1dy"]] = vehicles[["x", "y"]]
        vehicles["j1ot"] = self.time
        vehicles["j1dt"] = self._NEVER

        vehicles["j2m"] = Jobs.NULL
        vehicles["j3m"] = Jobs.NULL

        null_loc_cols = ["j2ox", "j2dx", "j3ox", "j3dx"]
        null_time_cols = ["j2ot", "j2dt", "j3ot", "j3dt"]
        vehicles[null_loc_cols] = self._NULL_X
        vehicles[["j2oy", "j2dy", "j3oy", "j3dy"]] = self._NULL_Y
        vehicles[null_time_cols] = self._NEVER

        vehicles["avail_t"] = self.time
        vehicles[["avail_x", "avail_y"]] = vehicles[["x", "y"]]
        vehicles["epoch_t"] = self._NEVER

        vehicles["battery"] = 100.0
        vehicles["capacity"] = self._agv_capacity
        vehicles["load"] = 0.0

        self._vehicles = vehicles.astype({
            'x': float, 'y': float,
            'j1m': int, 'j2m': int, 'j3m': int,
            'j1ox': float, 'j1oy': float, 'j1dx': float, 'j1dy': float,
            'j2ox': float, 'j2oy': float, 'j2dx': float, 'j2dy': float,
            'j3ox': float, 'j3oy': float, 'j3dx': float, 'j3dy': float,
            'j1ot': float, 'j1dt': float, 'j2ot': float, 'j2dt': float, 'j3ot': float, 'j3dt': float,
            'avail_t': float, 'avail_x': float, 'avail_y': float,
            'epoch_t': float, 'battery': float, 'capacity': float, 'load': float
        })

    def _generate_requests(self) -> None:
        """Generates dynamic transport requests using Poisson process."""
        lambda_rate = self._num_requests / self._MAX_TIME
        inter_arrivals = self._request_sampler.exponential(scale=1.0 / lambda_rate, size=self._num_requests * 2)
        arrival_times = np.cumsum(inter_arrivals)
        arrival_times = arrival_times[arrival_times < self._MAX_TIME]

        n_reqs = len(arrival_times)
        pd_station_indices = self._geom.pd_stations.index.to_numpy()

        origin_indices = self._request_sampler.choice(pd_station_indices, size=n_reqs)
        destination_indices = np.zeros_like(origin_indices)
        for i in range(n_reqs):
            valid_dest = pd_station_indices[pd_station_indices != origin_indices[i]]
            destination_indices[i] = self._request_sampler.choice(valid_dest)

        origin_locs = self._geom.stations.loc[origin_indices, ['x', 'y']].to_numpy()
        dest_locs = self._geom.stations.loc[destination_indices, ['x', 'y']].to_numpy()

        distances = self._geom.dist(origin_locs, dest_locs, pairwise=False)
        processing_times = distances / self._travel_speed
        deadlines = arrival_times + processing_times + self._time_window_slack

        requests = pd.DataFrame({
            'ox': origin_locs[:, 0],
            'oy': origin_locs[:, 1],
            'dx': dest_locs[:, 0],
            'dy': dest_locs[:, 1],
            'time': arrival_times,
            'pt': processing_times,
            'deadline': deadlines,
            'weight': np.full(n_reqs, self._request_weight),
            'cost_rate': np.full(n_reqs, self._tardiness_cost_rate),
            'released': np.zeros(n_reqs, dtype=bool),
            'rejected': np.zeros(n_reqs, dtype=bool),
            'completed': np.zeros(n_reqs, dtype=bool),
            'vehicle': [pd.NA] * n_reqs,
            'delivery_time': [np.nan] * n_reqs,
            'tardiness': np.zeros(n_reqs, dtype=float),
        })

        # Sentinel dummy request at _NEVER
        dummy_df = pd.DataFrame([{
            'ox': self._NULL_X, 'oy': self._NULL_Y,
            'dx': self._NULL_X, 'dy': self._NULL_Y,
            'time': self._NEVER, 'pt': 0.0, 'deadline': self._NEVER,
            'weight': 0.0, 'cost_rate': 0.0,
            'released': False, 'rejected': False, 'completed': False,
            'vehicle': pd.NA, 'delivery_time': np.nan, 'tardiness': 0.0
        }])

        self._requests = pd.concat([requests, dummy_df], ignore_index=True).sort_values(by="time").reset_index(drop=True)
        self._requests = self._requests.astype({
            'ox': float, 'oy': float, 'dx': float, 'dy': float,
            'time': float, 'pt': float, 'deadline': float, 'weight': float,
            'cost_rate': float, 'released': bool, 'rejected': bool, 'completed': bool,
            'vehicle': object, 'delivery_time': float, 'tardiness': float
        })

    def get_noop_action(self) -> Dict[str, np.ndarray]:
        """Returns no-operation action."""
        return {
            "req_rejections": np.zeros(self.num_pending_requests, dtype=int),
            "req_assgts": np.full(self.num_pending_requests, fill_value=self._V, dtype=int),
            "reposition": np.full(self._V, fill_value=self._D, dtype=int)
        }

    def get_random_action(self) -> Dict[str, np.ndarray]:
        """Returns random valid action."""
        act = self.action_space.sample()
        if "req_assgts" in act and len(act["req_assgts"]) > 0:
            assigned = act["req_assgts"][act["req_assgts"] != self._V]
            _, unique_idx = np.unique(assigned, return_index=True)
            mask = np.zeros_like(act["req_assgts"], dtype=bool)
            orig_indices = np.where(act["req_assgts"] != self._V)[0][unique_idx]
            mask[orig_indices] = True
            act["req_assgts"] = np.where(mask, act["req_assgts"], self._V)
        return act
