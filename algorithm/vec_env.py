import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pydispatching.agv_env import AGVEnv


class EnvFactory:
    """
    Picklable factory for instantiating AGVEnv instances inside Windows multiprocessing workers.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self) -> AGVEnv:
        return AGVEnv(**self.kwargs)


def _vec_worker(remote, parent_remote, env_factory: EnvFactory):
    parent_remote.close()
    env = env_factory()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, info = env.step(data)
                if terminated:
                    info["terminal_info"] = {
                        "total_cost": env.total_cost,
                        "tardiness_cost": env.total_tardiness_cost,
                        "travel_distance": env.total_travel_distance,
                        "completed_requests": env.num_completed_requests,
                        "stranded_agvs": env.num_stranded_vehicles,
                        "num_requests": len(env._requests) - 1,
                        "num_vehicles": env._V,
                        "rewards": env.rewards
                    }
                    obs = env.reset()
                remote.send((obs, reward, terminated, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except EOFError:
        pass
    except Exception as e:
        remote.send(("error", str(e)))
    finally:
        try:
            env.close()
        except:
            pass


class SubprocVecAGVEnv:
    """
    High-performance multiprocessing vectorized environment for AGV-Gym.
    Runs multiple AGV-Gym simulation instances across multi-core CPUs (e.g. AMD Ryzen 9 8940HX).
    """

    def __init__(self, env_factories: List[EnvFactory]):
        self.num_envs = len(env_factories)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        ctx = mp.get_context("spawn")
        for work_remote, remote, factory in zip(self.work_remotes, self.remotes, env_factories):
            p = ctx.Process(target=_vec_worker, args=(work_remote, remote, factory))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self) -> List[Dict[str, np.ndarray]]:
        """Resets all worker environments simultaneously."""
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Steps all worker environments concurrently."""
        for remote, action in zip(self.remotes, actions_list):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, dones, infos = zip(*results)
        return list(obs_list), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def close(self):
        """Closes all worker processes."""
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except:
                pass
        for p in self.processes:
            p.join(timeout=1.0)
