import unittest
import torch
import numpy as np
import os
import glob
import shutil
from pydispatching.agv_env import AGVEnv
from algorithm.dqn import DQNObsEncoder, DQNNetwork, ReplayBuffer, DQNTrainer


class TestDQN(unittest.TestCase):
    """
    Unit test suite for Deep Q-Network (DQN) on AGV-Gym.
    Verifies state encoder, network architectures (Standard & Dueling), action masking,
    replay buffer, training step, TensorBoard monitoring, and policy evaluation.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n=======================================================")
        print(f"Running DQN Unit Tests on Device: {cls.device}")
        if cls.device.type == "cuda":
            print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"=======================================================\n")

    def setUp(self):
        self.env = AGVEnv(
            num_vehicles=9,
            num_requests=10,
            alpha=0.5,
            seed=42,
            max_interdecision_time=120.0
        )
        self.encoder = DQNObsEncoder(self.env, max_req_slots=15)
        self.test_log_dir = "runs/unittest_dqn"

    def tearDown(self):
        if os.path.exists(self.test_log_dir):
            try:
                shutil.rmtree(self.test_log_dir)
            except Exception:
                pass

    def test_obs_encoder(self):
        """Verify observation encoder outputs flat state vector and correct mask shapes."""
        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)

        expected_dim = 1 + (9 * 6) + (15 * 9)
        self.assertEqual(flat_obs.shape, (expected_dim,))
        self.assertEqual(req_mask.shape, (15, 10))
        self.assertEqual(repos_mask.shape, (9, 10))
        self.assertEqual(flat_obs.device.type, self.device.type)
        self.assertIsInstance(num_actual, int)
        self.assertIsInstance(total_pending, int)

    def test_dueling_dqn_network_forward(self):
        """Verify Dueling DQN forward pass output shapes for unbatched and batched inputs."""
        model = DQNNetwork(
            obs_dim=self.encoder.obs_dim,
            num_vehicles=9,
            max_req_slots=15,
            num_charging_stations=9,
            hidden_dim=128,
            dueling=True
        ).to(self.device)

        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, _, _ = self.encoder.encode(obs, self.device)

        # Unbatched forward
        q_assign, q_repos = model.forward(flat_obs, req_mask, repos_mask)
        self.assertEqual(q_assign.shape, (15, 10))
        self.assertEqual(q_repos.shape, (9, 10))

        # Batched forward
        batch_obs = torch.stack([flat_obs, flat_obs])
        batch_req_m = torch.stack([req_mask, req_mask])
        batch_repos_m = torch.stack([repos_mask, repos_mask])
        b_q_assign, b_q_repos = model.forward(batch_obs, batch_req_m, batch_repos_m)
        self.assertEqual(b_q_assign.shape, (2, 15, 10))
        self.assertEqual(b_q_repos.shape, (2, 9, 10))

    def test_standard_dqn_network_forward(self):
        """Verify Standard (non-dueling) DQN forward pass output shapes."""
        model = DQNNetwork(
            obs_dim=self.encoder.obs_dim,
            num_vehicles=9,
            max_req_slots=15,
            num_charging_stations=9,
            hidden_dim=128,
            dueling=False
        ).to(self.device)

        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, _, _ = self.encoder.encode(obs, self.device)
        q_assign, q_repos = model.forward(flat_obs, req_mask, repos_mask)

        self.assertEqual(q_assign.shape, (15, 10))
        self.assertEqual(q_repos.shape, (9, 10))

    def test_action_selection_and_env_step(self):
        """Verify action selection under epsilon exploration and greedy mode produces valid AGVEnv actions."""
        trainer = DQNTrainer(
            env=self.env, max_req_slots=15, hidden_dim=128, device=self.device
        )
        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)

        # 1. Greedy action selection
        act_greedy = trainer.select_action(
            flat_obs, req_mask, repos_mask, num_actual, total_pending, epsilon=0.0, deterministic=True
        )
        action_dict = act_greedy["action_dict"]
        self.assertIn("req_assgts", action_dict)
        self.assertIn("reposition", action_dict)
        self.assertIn("req_rejections", action_dict)
        self.assertEqual(len(action_dict["req_assgts"]), total_pending)
        self.assertEqual(len(action_dict["reposition"]), 9)

        # 2. Step environment
        next_obs, reward, terminal, info = self.env.step(action_dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminal, bool)

        # 3. Epsilon random action selection
        act_random = trainer.select_action(
            flat_obs, req_mask, repos_mask, num_actual, total_pending, epsilon=1.0, deterministic=False
        )
        self.assertEqual(len(act_random["action_dict"]["req_assgts"]), total_pending)

    def test_replay_buffer(self):
        """Verify ReplayBuffer stores transitions and samples correct tensor batch shapes."""
        buffer = ReplayBuffer(capacity=100)
        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)
        req_assgts = torch.zeros(15, dtype=torch.long, device=self.device)
        repos_raw = torch.zeros(9, dtype=torch.long, device=self.device)

        for _ in range(20):
            buffer.add(
                flat_obs=flat_obs,
                req_mask=req_mask,
                repos_mask=repos_mask,
                num_actual=num_actual,
                req_assgts=req_assgts,
                repos_raw=repos_raw,
                reward=-10.0,
                next_flat_obs=flat_obs,
                next_req_mask=req_mask,
                next_repos_mask=repos_mask,
                next_num_actual=num_actual,
                done=False
            )

        self.assertEqual(len(buffer), 20)
        batch = buffer.sample(batch_size=8, device=self.device)
        self.assertEqual(batch["flat_obs"].shape, (8, self.encoder.obs_dim))
        self.assertEqual(batch["req_mask"].shape, (8, 15, 10))
        self.assertEqual(batch["repos_mask"].shape, (8, 9, 10))
        self.assertEqual(batch["rewards"].shape, (8,))
        self.assertEqual(batch["dones"].shape, (8,))

    def test_dqn_train_step_and_tensorboard(self):
        """Verify DQNTrainer training gradient updates and TensorBoard logging."""
        trainer = DQNTrainer(
            env=self.env,
            max_req_slots=15,
            hidden_dim=128,
            batch_size=8,
            tensorboard=True,
            log_dir=self.test_log_dir,
            device=self.device
        )

        obs = self.env.reset()
        flat_obs, req_mask, repos_mask, num_actual, total_pending = self.encoder.encode(obs, self.device)
        act_out = trainer.select_action(flat_obs, req_mask, repos_mask, num_actual, total_pending)

        # Seed replay buffer
        for _ in range(16):
            trainer.buffer.add(
                flat_obs=flat_obs,
                req_mask=req_mask,
                repos_mask=repos_mask,
                num_actual=num_actual,
                req_assgts=act_out["req_assgts_tensor"],
                repos_raw=act_out["repos_raw_tensor"],
                reward=-5.0,
                next_flat_obs=flat_obs,
                next_req_mask=req_mask,
                next_repos_mask=repos_mask,
                next_num_actual=num_actual,
                done=False
            )

        train_stats = trainer.train_step()
        self.assertIn("loss", train_stats)
        self.assertIn("q_mean", train_stats)
        self.assertIn("td_error", train_stats)
        self.assertIn("epsilon", train_stats)
        print(f"DQN Train Step Result: {train_stats}")

        # Check TensorBoard writer created event files
        if trainer.writer is not None:
            trainer.writer.flush()
            event_files = glob.glob(os.path.join(self.test_log_dir, "events.out.tfevents.*"))
            self.assertGreater(len(event_files), 0, "TensorBoard event files should be created.")

    def test_dqn_short_training_and_evaluation(self):
        """Verify full DQN train() and evaluate() pipeline execution."""
        trainer = DQNTrainer(
            env=self.env,
            max_req_slots=15,
            hidden_dim=128,
            batch_size=8,
            tensorboard=False,
            device=self.device
        )

        train_res = trainer.train(total_steps=16, warmup_steps=4, train_freq=2)
        self.assertEqual(train_res["total_steps"], 16)
        self.assertIn("mean_loss", train_res)

        eval_res = trainer.evaluate(num_episodes=1, render=False, deterministic=True)
        self.assertIn("mean_cost", eval_res)
        self.assertIn("mean_dist", eval_res)
        self.assertIn("mean_completed", eval_res)
        print(f"DQN Short Eval Result: {eval_res}")

    def test_dqn_checkpoint_save_and_load(self):
        """Verify full DQN checkpoint saving, restoring, and state validation."""
        import tempfile
        trainer = DQNTrainer(
            env=self.env,
            max_req_slots=15,
            hidden_dim=128,
            batch_size=8,
            tensorboard=False,
            device=self.device
        )
        trainer.total_env_steps = 543
        trainer.total_train_updates = 42
        trainer.epsilon = 0.35

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_dqn.ckpt")
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(os.path.isfile(ckpt_path))

            # Create new trainer and load
            new_trainer = DQNTrainer(
                env=self.env,
                max_req_slots=15,
                hidden_dim=128,
                batch_size=8,
                tensorboard=False,
                device=self.device
            )
            loaded_ckpt = new_trainer.load_checkpoint(ckpt_path, load_optimizer=True)

            self.assertEqual(new_trainer.total_env_steps, 543)
            self.assertEqual(new_trainer.total_train_updates, 42)
            self.assertAlmostEqual(new_trainer.epsilon, 0.35, places=4)
            self.assertEqual(loaded_ckpt["algorithm"], "DQN")

    def test_vec_dqn_multiprocessing(self):
        """Verify multi-worker vectorized DQN experience collection and training step."""
        from algorithm.vec_env import SubprocVecAGVEnv, EnvFactory
        factories = [
            EnvFactory(num_vehicles=9, num_requests=10, alpha=0.5, seed=200 + i, max_interdecision_time=120.0)
            for i in range(2)
        ]
        vec_env = SubprocVecAGVEnv(factories)
        try:
            trainer = DQNTrainer(
                env=self.env,
                vec_env=vec_env,
                max_req_slots=15,
                hidden_dim=128,
                batch_size=8,
                utd_ratio=1,
                tensorboard=False,
                device=self.device
            )
            train_res = trainer.train(total_steps=16, warmup_steps=4)
            self.assertGreaterEqual(train_res["total_steps"], 16)
        finally:
            vec_env.close()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    unittest.main()
