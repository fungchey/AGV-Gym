import matplotlib.pyplot as plt
from pydispatching import AGVEnv
from algorithm import heuristics
import numpy as np
import time


def main(render: bool = False, num_eval_episodes: int = 3, fleet_size: int = 9, num_requests: int = 900, alpha: float = 0.9):
    """
    Main evaluation script for AGV-Gym using the Nearest Neighbour baseline.
    """
    env_params = {
        "num_vehicles": fleet_size,
        "num_requests": num_requests,
        "alpha": alpha,
        "stochastic": False,
        "seed": 42,
        "action_timelimit": np.inf,
        "max_interdecision_time": 60,
        "for_evaluation": True,
        "nickname": "agv_nn_eval"
    }
    
    print(f"Initializing AGV-Gym with fleet_size={fleet_size}, requests={num_requests}, alpha={alpha}...")
    env = AGVEnv(**env_params) 
    heu = heuristics.Heuristics(env)

    all_eps_rewards = []
    all_eps_costs = []
    all_eps_distances = []
    all_eps_tardiness = []
    all_eps_completed = []

    for episode in range(num_eval_episodes): 
        start_time = time.time()
        obs = env.reset()
        terminal = False
        step_count = 0
        total_reqs = len(env._requests) - 1

        if render:
            plt.figure("AGV-Gym Simulation", figsize=(12, 7))
            rgb = env.render()
            im = env.draw_label(rgb, episode, 0.0)
            plt.imshow(im)
            plt.ion()
            plt.show()

        while not terminal:
            # Nearest Neighbour with proactive charging
            action = heu.nearest_neighbour(obs)

            next_obs, new_rwd, terminal, info = env.step(action)
            obs = next_obs
            step_count += 1

            if render and step_count % 10 == 0:
                rgb = env.render()
                im = env.draw_label(rgb, episode, env.rewards, info=info)
                plt.clf()
                plt.imshow(im)
                plt.pause(0.001)

        elapsed = time.time() - start_time
        comp_pct = min(100.0, (env.num_completed_requests / max(1, total_reqs)) * 100.0)
        stranded_count = env.num_stranded_vehicles
        stranded_pct = (stranded_count / env._V) * 100.0

        print(f"Episode {episode + 1}/{num_eval_episodes} completed in {elapsed:.2f}s ({step_count} steps):", flush=True)
        print(f"  - Total Cost: {env.total_cost:.2f}", flush=True)
        print(f"  - Tardiness Cost: {env.total_tardiness_cost:.2f}", flush=True)
        print(f"  - Total Travel Distance: {env.total_travel_distance:.2f}", flush=True)
        print(f"  - Completed Requests: {env.num_completed_requests}/{total_reqs} ({comp_pct:.1f}%)", flush=True)
        print(f"  - Stranded AGVs (Depleted Battery): {stranded_count}/{env._V} ({stranded_pct:.1f}%)", flush=True)
        print(f"  - Mean Fleet Battery: {env._vehicles['battery'].mean():.1f}% (min: {env._vehicles['battery'].min():.1f}%)", flush=True)

        all_eps_rewards.append(env.rewards)
        all_eps_costs.append(env.total_cost)
        all_eps_distances.append(env.total_travel_distance)
        all_eps_tardiness.append(env.total_tardiness_cost)
        all_eps_completed.append(env.num_completed_requests)
        all_eps_stranded = getattr(locals(), 'all_eps_stranded', [])
        if 'all_eps_stranded' not in locals() or len(all_eps_stranded) != len(all_eps_costs) - 1:
            all_eps_stranded = []
        all_eps_stranded.append(stranded_count)

    print("\n" + "=" * 50, flush=True)
    print("EVALUATION SUMMARY OVER ALL EPISODES:", flush=True)
    print(f"  - Mean Total Cost: {np.mean(all_eps_costs):.2f} +/- {np.std(all_eps_costs):.2f}", flush=True)
    print(f"  - Mean Reward: {np.mean(all_eps_rewards):.2f}", flush=True)
    print(f"  - Mean Tardiness Cost: {np.mean(all_eps_tardiness):.2f}", flush=True)
    print(f"  - Mean Distance Travelled: {np.mean(all_eps_distances):.2f}", flush=True)
    print(f"  - Mean Completed Requests: {np.mean(all_eps_completed):.1f}/{total_reqs} ({(np.mean(all_eps_completed)/max(1, total_reqs))*100.0:.1f}%)", flush=True)
    print(f"  - Mean Stranded AGVs: {np.mean(all_eps_stranded):.1f}/{env._V} ({(np.mean(all_eps_stranded)/env._V)*100.0:.1f}%)", flush=True)
    print("=" * 50 + "\n", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AGV-Gym Evaluation Script with Baseline Heuristics")
    parser.add_argument("--render", action="store_true", help="Render simulation visualization")
    parser.add_argument("--num_eval_episodes", "--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--fleet_size", type=int, default=9, help="Fleet size (e.g. 9, 12, 15, 18)")
    parser.add_argument("--num_requests", "--requests", type=int, default=900, help="Number of requests (e.g. 450, 900, 1800)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha weighting coefficient [0.0 - 1.0] (default: 0.9)")

    args, unknown = parser.parse_known_args()
    render_val = args.render

    # Also handle key=value style arguments (e.g., render=true, fleet_size=12)
    for u in unknown:
        if "=" in u:
            k, v = u.lower().split("=", 1)
            if k in ["render", "--render"]:
                render_val = v in ["true", "1", "yes", "y", "t"]
            elif k in ["fleet_size", "--fleet_size", "num_vehicles"]:
                args.fleet_size = int(v)
            elif k in ["num_requests", "--num_requests", "requests"]:
                args.num_requests = int(v)
            elif k in ["alpha", "--alpha"]:
                args.alpha = float(v)
            elif k in ["num_eval_episodes", "--num_eval_episodes", "episodes"]:
                args.num_eval_episodes = int(v)
        elif u.lower() in ["render", "render=true"]:
            render_val = True

    main(
        render=render_val,
        num_eval_episodes=args.num_eval_episodes,
        fleet_size=args.fleet_size,
        num_requests=args.num_requests,
        alpha=args.alpha
    )
