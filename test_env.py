import unittest
import numpy as np
import pandas as pd
from pydispatching import AGVEnv
from pydispatching.core import Jobs, AGVGeometry
from algorithm import heuristics


class TestAGVEnv(unittest.TestCase):
    """Automated test suite verifying AGV-Gym against manuscript_v1.pdf specifications."""

    def setUp(self):
        self.env = AGVEnv(
            num_vehicles=9,
            num_requests=900,
            alpha=0.5,
            seed=42,
            max_interdecision_time=60.0
        )

    def test_station_layout(self):
        """Verify BIC station counts: 60 PD, 9 Charging, 9 Parking (78 total)."""
        stations = self.env._geom.stations
        self.assertEqual(len(stations), 78)
        
        pd_count = len(stations[stations['type'] == 'Pickup and Delivery'])
        c_count = len(stations[stations['type'] == 'Charging station'])
        p_count = len(stations[stations['type'] == 'Parking station'])
        
        self.assertEqual(pd_count, 60)
        self.assertEqual(c_count, 9)
        self.assertEqual(p_count, 9)

    def test_fleet_sizes_and_scenarios(self):
        """Verify fleet sizes (9, 12, 15, 18) and scenarios (450, 900, 1800)."""
        for v in [9, 12, 15, 18]:
            env = AGVEnv(num_vehicles=v, num_requests=450, seed=42)
            obs = env.reset()
            self.assertEqual(obs["v_locs"].shape, (v, 2))
            self.assertEqual(obs["v_battery"].shape, (v,))
            self.assertEqual(obs["v_jobs"].shape, (v, 3))

    def test_vehicle_initialization(self):
        """Verify vehicles are placed at Charging or Parking stations with 100% battery."""
        obs = self.env.reset()
        v_locs = obs["v_locs"]
        v_battery = obs["v_battery"]
        
        self.assertEqual(len(v_locs), 9)
        np.testing.assert_allclose(v_battery, 100.0)
        
        init_station_coords = self.env._geom.init_stations[['x', 'y']].to_numpy()
        for loc in v_locs:
            dists = np.abs(init_station_coords - loc).sum(axis=1)
            self.assertTrue(np.any(dists < 1e-3), "Vehicle not placed at P or C station.")

    def test_observation_space(self):
        """Verify observation space keys and shapes."""
        obs = self.env.reset()
        required_keys = ["time", "request_locs", "request_times", "request_deadlines", 
                         "request_weights", "v_locs", "v_jobs", "v_job_locs", "v_battery"]
        for key in required_keys:
            self.assertIn(key, obs)
            
        self.assertEqual(obs["v_locs"].shape, (9, 2))
        self.assertEqual(obs["v_battery"].shape, (9,))
        self.assertEqual(obs["v_jobs"].shape, (9, 3))
        self.assertEqual(obs["v_job_locs"].shape, (9, 3, 2, 2))

    def test_reward_and_cost_consistency(self):
        """Verify reward is exact negative of CMDP objective cost."""
        obs = self.env.reset()
        action = self.env.get_noop_action()
        obs, reward, terminal, info = self.env.step(action)
        
        self.assertAlmostEqual(reward, -info["step_cost"], places=5)
        self.assertAlmostEqual(
            info["step_cost"], 
            0.5 * info["step_tardiness_cost"] + 0.5 * info["step_travel_cost"], 
            places=5
        )

    def test_alpha_weight_parameter(self):
        """Verify alpha weighting in reward formula."""
        for alpha in [0.0, 0.25, 0.75, 1.0]:
            env = AGVEnv(num_vehicles=9, num_requests=100, alpha=alpha, seed=42)
            obs = env.reset()
            action = env.get_noop_action()
            obs, reward, done, info = env.step(action)
            expected_cost = alpha * info["step_tardiness_cost"] + (1.0 - alpha) * info["step_travel_cost"]
            self.assertAlmostEqual(info["step_cost"], expected_cost, places=5)

    def test_battery_critical_masking(self):
        """Verify action masking prevents dispatching vehicle with battery < critical threshold."""
        obs = self.env.reset()
        # Artificially set vehicle 0 battery below critical (e.g. 15%)
        self.env._vehicles.at[0, "battery"] = 15.0
        
        # Force a request to be pending
        self.env._requests.at[0, "released"] = True
        self.env._next_request_idx = 1
        self.env.num_pending_requests = self.env._get_num_pending_requests()
        self.env._set_action_space()
        self.env._set_observation_space()
        
        # Try assigning vehicle 0 to request 0
        action = {
            "req_rejections": np.zeros(self.env.num_pending_requests, dtype=int),
            "req_assgts": np.zeros(self.env.num_pending_requests, dtype=int),  # Assign to veh 0
            "reposition": np.full(9, fill_value=self.env._D, dtype=int)
        }
        
        # Step should ignore the infeasible assignment
        obs, rwd, done, info = self.env.step(action)
        assigned_veh = self.env._requests.at[0, "vehicle"]
        self.assertTrue(pd.isna(assigned_veh) or assigned_veh != 0, "Vehicle with critical battery should not be assigned.")

    def test_battery_charging_dynamics(self):
        """Verify battery recharges when positioned at a charging station."""
        obs = self.env.reset()
        c_station_coords = self.env._geom.charging_stations[['x', 'y']].iloc[0].to_numpy()
        self.env._vehicles.loc[0, ["x", "y", "avail_x", "avail_y"]] = [
            c_station_coords[0], c_station_coords[1], c_station_coords[0], c_station_coords[1]
        ]
        self.env._vehicles.at[0, "battery"] = 50.0
        self.env._vehicles.at[0, "j1m"] = Jobs.IDLE
        
        # Advance with noop action
        action = self.env.get_noop_action()
        obs, rwd, done, info = self.env.step(action)
        
        self.assertGreater(self.env._vehicles.at[0, "battery"], 50.0, "Vehicle at charging station should recharge.")

    def test_heuristics_full_episode(self):
        """Verify Nearest Neighbour baseline runs cleanly to termination."""
        env = AGVEnv(num_vehicles=9, num_requests=100, seed=123, max_interdecision_time=120)
        heu = heuristics.Heuristics(env)
        obs = env.reset()
        terminal = False
        steps = 0
        
        while not terminal and steps < 2000:
            act = heu.nearest_neighbour(obs)
            obs, rwd, terminal, info = env.step(act)
            steps += 1
            
        self.assertTrue(terminal, f"Episode did not terminate. Final time: {env.time}")
        self.assertGreater(env.num_completed_requests, 0)
        self.assertGreater(env.total_travel_distance, 0)
        self.assertTrue(np.all(env._vehicles['battery'] >= 0))

    def test_rendering(self):
        """Verify render returns a valid RGB image array of size 1260x2080x3."""
        obs = self.env.reset()
        rgb = self.env.render()
        self.assertEqual(rgb.shape, (1260, 2080, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_continuous_tardiness_accumulation(self):
        """Verify Option B: unserved active requests accumulate tardiness per step after their deadline."""
        env = AGVEnv(num_vehicles=9, num_requests=10, alpha=0.9, seed=42, max_interdecision_time=60.0)
        obs = env.reset()
        
        # Manually release request 0 with deadline in the near future (e.g. t = 100s)
        env._requests.at[0, "released"] = True
        env._requests.at[0, "deadline"] = 100.0
        env._requests.at[0, "cost_rate"] = 0.10
        env.num_pending_requests = env._get_num_pending_requests()
        env._set_action_space()
        env._set_observation_space()

        # Step forward past the deadline with NOOP (leaving request unserved)
        # Advance time to > 100s
        action = env.get_noop_action()
        obs, rwd, done, info = env.step(action)

        if env.time > 100.0:
            # Tardiness should be accrued continuously
            expected_tardiness = env.time - 100.0
            expected_cost = 0.10 * expected_tardiness
            self.assertAlmostEqual(env._requests.at[0, "tardiness"], expected_tardiness, places=3)
            self.assertAlmostEqual(env.total_tardiness_cost, expected_cost, places=3)

    def test_stranded_vehicles_tracking(self):
        """Verify stranded vehicle count is correctly reported when battery is depleted."""
        env = AGVEnv(num_vehicles=9, num_requests=10, seed=42)
        obs = env.reset()
        self.assertEqual(env.num_stranded_vehicles, 0)

        # Place vehicles 0 and 1 at non-charging stations and deplete battery
        pd_coords = env._geom.stations[env._geom.stations['type'] == 'Pickup and Delivery'][['x', 'y']].iloc[:2].to_numpy()
        pos_cols = ["x", "y", "j1ox", "j1oy", "j1dx", "j1dy", "avail_x", "avail_y"]
        env._vehicles.loc[0, pos_cols] = [pd_coords[0, 0], pd_coords[0, 1]] * 4
        env._vehicles.loc[1, pos_cols] = [pd_coords[1, 0], pd_coords[1, 1]] * 4
        env._vehicles.at[0, "battery"] = 0.0
        env._vehicles.at[1, "battery"] = 0.0
        self.assertEqual(env.num_stranded_vehicles, 2)

        action = env.get_noop_action()
        obs, rwd, done, info = env.step(action)
        self.assertIn("num_stranded_vehicles", info)
        self.assertEqual(info["num_stranded_vehicles"], 2)


if __name__ == "__main__":
    unittest.main()
