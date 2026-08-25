import numpy as np
import scipy.spatial.distance as distance
import logging
from typing import Dict, List, Tuple
from pydispatching.agv_env import AGVEnv
from pydispatching.core import Jobs


class Heuristics:
    """
    Heuristics baseline for AGV scheduling with battery constraints (ASP-BC).
    Implements Nearest-Neighbour dispatching with proactive battery charging management.
    """

    def __init__(self, env: AGVEnv):
        self.env = env
        self.num_vehicles = env.num_vehicles
        self.num_stations = env.num_stations
        self._geom = env._geom
        self.critical_battery_level = float(env._params.get('Critical_battery_level', {}).get('level', 20.0))
        self.non_critical_battery_level = float(env._params.get('Non_critical_battery_level', {}).get('level', 80.0))
        self.travel_discharge_rate = float(env._params.get('Travel_discharging_rate', {}).get('rate', 0.0055))
        self.travel_speed = float(env._params.get('Travel_speed', {}).get('speed', 1.0))
        
        # Precompute charging station coordinates and indices
        self.c_locs = self._geom.charging_stations[['x', 'y']].to_numpy()
        self.c_indices = self._geom.charging_stations.index.to_numpy()

    def nearest_neighbour(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes nearest-neighbour action dictionary from current observation.
        """
        self.current_obs = obs
        self.current_time = float(obs["time"][0])
        num_pending = len(obs["request_times"])

        v_locs = obs["v_locs"]
        v_jobs = obs["v_jobs"]
        v_battery = obs["v_battery"]

        # 1. Identify vehicle availability
        job_avail = np.isin(v_jobs[:, 0], (Jobs.IDLE, Jobs.REPOSITION, Jobs.CHARGING, Jobs.NULL))
        batt_avail = v_battery > self.critical_battery_level
        avail_mask = job_avail & batt_avail
        avail_veh_indices = np.where(avail_mask)[0]

        assignments = {}  # {req_idx: assigned_veh_idx}
        assigned_vehicles = set()

        if num_pending > 0 and len(avail_veh_indices) > 0:
            req_times = obs["request_times"]
            sorted_req_indices = np.argsort(req_times)
            req_locs = obs["request_locs"]  # (num_pending, 2, 2)

            for req_idx in sorted_req_indices:
                if len(assigned_vehicles) == len(avail_veh_indices):
                    break

                req_origin = req_locs[req_idx, 0, :]
                req_dest = req_locs[req_idx, 1, :]

                # Filter candidate available vehicles not yet assigned
                candidate_v_indices = [v for v in avail_veh_indices if v not in assigned_vehicles]
                if not candidate_v_indices:
                    break

                cand_locs = v_locs[candidate_v_indices]
                cand_batteries = v_battery[candidate_v_indices]

                # Distance from request origin to candidates
                dists_to_origin = np.abs(cand_locs - req_origin).sum(axis=1)
                sorted_ranks = np.argsort(dists_to_origin)

                d_origin_dest = np.abs(req_origin - req_dest).sum()

                # Distance from destination to nearest charging station
                dist_to_charger = np.min(np.abs(self.c_locs - req_dest).sum(axis=1))
                charger_discharge = self.travel_discharge_rate * (dist_to_charger / self.travel_speed)

                for rk in sorted_ranks:
                    v_idx = candidate_v_indices[rk]
                    curr_bat = cand_batteries[rk]
                    d_to_orig = dists_to_origin[rk]

                    total_task_dist = d_to_orig + d_origin_dest
                    task_discharge = self.travel_discharge_rate * (total_task_dist / self.travel_speed)

                    if curr_bat - task_discharge - charger_discharge >= self.critical_battery_level:
                        assignments[req_idx] = v_idx
                        assigned_vehicles.add(v_idx)
                        break

        # 2. Build request assignments
        req_assgts = np.full(num_pending, fill_value=self.env._V, dtype=int)
        for req_idx, v_idx in assignments.items():
            req_assgts[req_idx] = v_idx

        # 3. Proactive Battery Charging / Repositioning
        reposition = np.full(self.num_vehicles, fill_value=self.env._D, dtype=int)
        busy_mask = np.isin(v_jobs[:, 0], (Jobs.SETUP, Jobs.PROCESS))
        newly_assigned_mask = np.isin(np.arange(self.num_vehicles), list(assignments.values()))
        cannot_repos = busy_mask | newly_assigned_mask

        # Fast distance computation to charging stations for all vehicles
        dist_to_chargers = distance.cdist(v_locs, self.c_locs, "cityblock")
        min_charger_dists = dist_to_chargers.min(axis=1)
        nearest_c_indices = self.c_indices[dist_to_chargers.argmin(axis=1)]

        for v_idx in range(self.num_vehicles):
            if cannot_repos[v_idx]:
                reposition[v_idx] = self.env._D
                continue

            v_bat = v_battery[v_idx]
            is_at_charger = min_charger_dists[v_idx] < 1e-3

            if not is_at_charger and v_bat <= self.non_critical_battery_level:
                reposition[v_idx] = nearest_c_indices[v_idx]
            else:
                reposition[v_idx] = self.env._D

        return {
            "req_rejections": np.zeros(num_pending, dtype=int),
            "req_assgts": req_assgts,
            "reposition": reposition
        }


heuristics = Heuristics
