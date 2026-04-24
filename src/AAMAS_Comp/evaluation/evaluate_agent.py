from time import perf_counter
from datetime import datetime
from AAMAS_Comp import base_agent
import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .benchmark import compute_benchmark_summary, print_benchmark_summary, save_benchmark_summary


def _default_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _infer_env_family(env) -> str:
    """Return a coarse environment family for benchmark performance shaping."""
    class_name = env.unwrapped.__class__.__name__.lower()
    if "frozenlake" in class_name:
        return "frozenlake"
    if "cartpole" in class_name:
        return "cartpole"
    return "other"


def _value_iteration_frozenlake(
    transitions,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> np.ndarray:
    """Solve for the converged optimal value function of the current FrozenLake MDP."""
    num_states = len(transitions)
    values = np.zeros(num_states, dtype=float)

    for _ in range(max_iterations):
        updated_values = values.copy()
        delta = 0.0

        for state, action_map in transitions.items():
            action_values = []
            for action in action_map:
                q_value = 0.0
                for prob, next_state, reward, terminated in action_map[action]:
                    future_value = 0.0 if terminated else gamma * values[next_state]
                    q_value += prob * (reward + future_value)
                action_values.append(q_value)

            if action_values:
                updated_values[state] = max(action_values)
                delta = max(delta, abs(updated_values[state] - values[state]))

        values = updated_values
        if delta < tolerance:
            break

    return values


def _compute_frozenlake_performance(env, obs) -> float:
    """Return normalized FrozenLake performance using the converged value function."""
    transitions = getattr(env, "P", None)
    if transitions is None:
        transitions = getattr(env.unwrapped, "P", None)
    if transitions is None:
        return 0.0

    state = int(obs["state"])
    values = _value_iteration_frozenlake(transitions)
    return float(np.clip(values[state], 0.0, 1.0))


def _compute_cartpole_performance(env, obs) -> float:
    """Return normalized CartPole performance using the IsaacGym-style reward."""
    state = np.asarray(obs["state"], dtype=float)
    if state.size < 4:
        return 0.0

    cart_pos, cart_vel, pole_angle, pole_vel = state[:4]
    reset_dist = float(getattr(env.unwrapped, "x_threshold", 2.4))

    reward = (
        1.0
        - pole_angle * pole_angle
        - 0.01 * abs(cart_vel)
        - 0.005 * abs(pole_vel)
    )

    if abs(cart_pos) > reset_dist or abs(pole_angle) > np.pi / 2:
        reward = -2.0

    reward_min = -2.0
    reward_max = 1.0
    normalized_reward = (reward - reward_min) / (reward_max - reward_min)
    return float(np.clip(normalized_reward, 0.0, 1.0))


def _compute_step_performance(env, obs) -> float | None:
    """Return the benchmark performance signal for the current decision state."""
    env_family = _infer_env_family(env)
    if env_family == "frozenlake":
        return _compute_frozenlake_performance(env, obs)
    if env_family == "cartpole":
        return _compute_cartpole_performance(env, obs)
    return None


def run_single_episode(env, agent, seed):
    """Runs single environment episode.

    Args:
        env (NSFrozenLakeWrapper): The gymnasium environment. Must be wrapped with NSFrozenLakeWrapper (or a subclass) to expose the necessary non-stationary interfaces like `get_planning_env()`.
        agent (Union[AAMAS_Comp.base_agent.ModelBasedAgent, AAMAS_Comp.base_agent.ModelFreeAgent]): Evaluation Agent.
        seed (int): Random number generator seed.

    Returns:
        dict: Per-step evaluation data including raw rewards, benchmark
        performance values, and disruption metadata.
    """

    obs, _ = env.reset(seed=seed)

    done = False
    truncated = False

    episode_metrics = {
        "step_number": [],
        "rewards": [],
        "performance": [],
        "observations": [],
        "notification": [],
        "actions": [],
        "decision_time": [],
        "info": [],
        "env_change": [],
        "disruption_step": [],
    }

    is_model_based_agent = isinstance(agent, base_agent.ModelBasedAgent)
    count = 0
    current_env_changed = 0

    while not (done or truncated):
        if current_env_changed and not episode_metrics["disruption_step"]:
            episode_metrics["disruption_step"].append(count)

        step_performance = _compute_step_performance(env, obs)

        if is_model_based_agent:
            planning_env = env.get_planning_env()
            action, decision_time = agent.validate_and_get_action(obs, planning_env)

        else:
            action, decision_time = agent.validate_and_get_action(obs, env.action_space)

        obs, reward, done, truncated, info = env.step(action)

        ground_truth_change = info.get("Ground Truth Env Change", {})
        if ground_truth_change:
            env_changed = int(max(ground_truth_change.values()))
        else:
            observed_change = obs.get("env_change", 0)
            if isinstance(observed_change, dict):
                env_changed = int(max(observed_change.values(), default=0))
            else:
                env_changed = int(bool(observed_change))

        if step_performance is None:
            step_performance = float(reward)

        episode_metrics["step_number"].append(count)
        episode_metrics["performance"].append(float(step_performance))
        episode_metrics["observations"].append(obs["state"])
        episode_metrics["notification"].append(obs["env_change"])
        episode_metrics["rewards"].append(reward)
        episode_metrics["actions"].append(action)
        episode_metrics["decision_time"].append(decision_time)
        episode_metrics["info"].append(info)
        episode_metrics["env_change"].append(env_changed)
        current_env_changed = env_changed
        count += 1

    return episode_metrics


def run_complete_evaluation(
    env,
    agent,
    start_seed,
    num_episodes,
    name_prefix,
    save_dir="results/",
    return_artifacts=False,
):
    """Runs multiple episodes with deterministic sequential seeding. Saves results as Compressed JSON file.
    Args:
        env: The gymnasium environment wrapped with NSFrozenLakeWrapper.
        agent: Evaluation Agent (ModelBasedAgent or ModelFreeAgent).
        start_seed (int): Starting seed. Each episode uses start_seed + i.
        num_episodes (int): Number of episodes to run.
        name_prefix (str): Experiment name prefix.
        save_dir (Path): path to save directory. Defaults to results.


    Returns:
        dict: Maps seed (str) to that episode's metrics dict.
    """
    results_table = {}

    start_time = perf_counter()

    for i in tqdm(range(num_episodes), desc=name_prefix):
        seed = start_seed + i
        episode_metrics = run_single_episode(env, agent, seed)
        results_table[str(seed)] = episode_metrics

    total_time = perf_counter() - start_time

    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    experiment_dir = save_dir / name_prefix
    os.makedirs(experiment_dir, exist_ok=True)

    metadata = {
        "name_prefix": name_prefix,
        "start_seed": start_seed,
        "end_seed": start_seed + num_episodes - 1,
        "num_episodes": num_episodes,
        "change_notification": env.change_notification, 
        "delta_change_notification": env.delta_change_notification,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        
    }

    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    json_path = experiment_dir / f"{name_prefix}.json"
    zip_path = experiment_dir / f"{name_prefix}.zip"

    with open(json_path, "w") as f:
        json.dump(results_table, f, default=_default_serializer)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, json_path.name)

    json_path.unlink()

    # Per-episode summary
    summary = {}
    for seed_key, ep in results_table.items():
        rewards = ep["rewards"]
        decision_times = ep["decision_time"]
        summary[seed_key] = {
            "total_reward": sum(rewards),
            "num_steps": len(rewards),
            "mean_decision_time": np.mean(decision_times),
            "std_decision_time": np.std(decision_times),
            "num_transition_fn_changes": sum(ep["env_change"]),
        }

    # Aggregate across all episodes
    all_rewards = [s["total_reward"] for s in summary.values()]
    all_steps = [s["num_steps"] for s in summary.values()]
    all_mean_dt = [s["mean_decision_time"] for s in summary.values()]
    all_tf_changes = [s["num_transition_fn_changes"] for s in summary.values()]

    aggregate = {
        "mean_total_reward": float(np.mean(all_rewards)),
        "std_total_reward": float(np.std(all_rewards)),
        "mean_episode_steps": float(np.mean(all_steps)),
        "mean_decision_time": float(np.mean(all_mean_dt)),
        "std_decision_time": float(np.std(all_mean_dt)),
        "mean_transition_fn_changes": float(np.mean(all_tf_changes)),
        "std_transition_fn_changes": float(np.std(all_tf_changes)),
    }

    summary_data = {
        "aggregate": aggregate,
        "per_episode": summary,
    }

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=_default_serializer)

    benchmark_data = compute_benchmark_summary(results_table)
    save_benchmark_summary(benchmark_data, experiment_dir / "benchmark.json")

    print(f"\n{'='*50}")
    print(f"Evaluation Summary: {name_prefix}")
    print(f"{'='*50}")
    print(f"Episodes:            {num_episodes}")
    print(f"Seed Range:          {start_seed} - {start_seed + num_episodes - 1}")
    print(f"Total Time:          {total_time:.2f}s")
    print(f"Mean Total Reward:   {aggregate['mean_total_reward']:.4f} +/- {aggregate['std_total_reward']:.4f}")
    print(f"Mean Episode Steps:  {aggregate['mean_episode_steps']:.1f}")
    print(f"Mean Decision Time:  {aggregate['mean_decision_time']:.6f}s +/- {aggregate['std_decision_time']:.6f}s")
    print(f"Mean Number of T(s,a) Changes: {aggregate['mean_transition_fn_changes']:.2f} +/- {aggregate['std_transition_fn_changes']:.2f}")
    print_benchmark_summary(name_prefix, benchmark_data)
    print(f"{'='*50}")
    print(f"Results saved to:    {experiment_dir}")

    if return_artifacts:
        return {
            "results_table": results_table,
            "summary": summary_data,
            "benchmark": benchmark_data,
            "metadata": metadata,
            "experiment_dir": experiment_dir,
        }

    return results_table
