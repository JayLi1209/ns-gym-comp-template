from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


EpisodeMetricFn = Callable[[dict[str, list[Any]]], float]


def _mean_or_default(values: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(np.asarray(values, dtype=float)))


def _weighted_mean_or_default(
    values: list[float],
    weights: list[float],
    default: float = 0.0,
) -> float:
    if not values:
        return float(default)

    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if np.sum(weights_array) <= 0:
        return float(np.mean(values_array))

    return float(np.average(values_array, weights=weights_array))


def compute_performance_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return total reward for one episode."""
    rewards = np.asarray(episode_metrics.get("rewards", []), dtype=float)
    return float(np.sum(rewards))


def compute_efficiency_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return mean decision time for one episode."""
    decision_times = np.asarray(episode_metrics.get("decision_time", []), dtype=float)
    if decision_times.size == 0:
        return 0.0
    return float(np.mean(decision_times))


def compute_resilience_metric(
    episode_metrics: dict[str, list[Any]],
    epsilon: float = 0.9,
    window: int = 5,
) -> float:
    """Compute the resilience score S_init * T_rec for one episode.

    Resilience measures how badly the agent stumbled immediately after a
    disruption (S_init) and how long it took to recover (T_rec). A lower
    product is better.

    Expected keys in episode_metrics
    ---------------------------------
    performance : list[float]
        Per-timestep performance signal p(t).
    disruption_step : list[int]
        Single-element list containing the disruption index t_d.
    """
    perf = np.asarray(episode_metrics.get("performance", []), dtype=float)
    disruption_steps = episode_metrics.get("disruption_step", [])

    if perf.size == 0 or not disruption_steps:
        return 1.0

    t_d = int(disruption_steps[0])
    if t_d <= 0 or t_d >= len(perf):
        return 1.0

    pre_start = max(0, t_d - window)
    eta = float(np.mean(perf[pre_start:t_d])) if t_d > pre_start else float(perf[t_d])

    if abs(eta) < 1e-8:
        eta = 1e-8

    p_at_disruption = perf[t_d] / abs(eta)
    s_init = float(np.clip(1.0 - p_at_disruption, 0.0, 1.0))

    recovery_threshold = epsilon * abs(eta)
    post_perf = perf[t_d:]
    t_rec = None

    for i in range(len(post_perf) - window + 1):
        window_slice = post_perf[i : i + window]
        if np.all(window_slice >= recovery_threshold):
            t_rec = float(i)
            break

    if t_rec is None:
        t_rec = float(len(post_perf))

    max_steps = float(max(len(post_perf), 1))
    t_rec_norm = float(np.clip(t_rec / max_steps, 0.0, 1.0))

    return float(s_init * t_rec_norm)


def compute_adaptability_metric(
    episode_metrics: dict[str, list[Any]],
    eta_star_fraction: float = 0.8,
    convergence_window: int = 10,
    convergence_beta: float = 0.9,
    stability_window: int = 5,
    stability_epsilon: float = 0.05,
    stability_xi: float = 1e-6,
) -> float:
    """Compute a scalar adaptability score for one episode.

    Adaptability combines convergence, post-shift stability, and convergence
    time into a single score where lower is better.

    Expected keys in episode_metrics
    ---------------------------------
    performance : list[float]
        Per-timestep performance signal p(t).
    disruption_step : list[int]
        Single-element list containing the disruption index t_d.
    """
    large_penalty = 1e6

    perf = np.asarray(episode_metrics.get("performance", []), dtype=float)
    disruption_steps = episode_metrics.get("disruption_step", [])

    if perf.size == 0 or not disruption_steps:
        return large_penalty

    t_d = int(disruption_steps[0])
    if t_d <= 0 or t_d >= len(perf):
        return large_penalty

    post_perf = perf[t_d:]
    t_post = len(post_perf)

    pre_start = max(0, t_d - convergence_window)
    eta_pre = float(np.mean(perf[pre_start:t_d])) if t_d > pre_start else float(perf[t_d])
    eta_star = eta_star_fraction * abs(eta_pre) if abs(eta_pre) > 1e-8 else eta_star_fraction

    converged = False
    t_conv_steps = t_post

    for i in range(t_post - convergence_window + 1):
        window_slice = post_perf[i : i + convergence_window]
        fraction_above = float(np.mean(window_slice >= eta_star))
        if fraction_above >= convergence_beta:
            converged = True
            t_conv_steps = i
            break

    c = 1 if converged else 0

    d_values: list[float] = []
    for i, t in enumerate(range(t_d, t_d + t_post)):
        lo = max(t_d, t - stability_window)
        hi = min(t_d + t_post - 1, t + stability_window)
        w_start = lo - t_d
        w_end = hi - t_d + 1
        local_mean = float(np.mean(post_perf[w_start:w_end]))

        p_t = float(post_perf[i])
        d_t = abs(p_t - local_mean) / (abs(local_mean) + stability_xi)
        d_values.append(d_t)

    d_arr = np.asarray(d_values, dtype=float)
    v_self = float(np.mean(np.maximum(d_arr - stability_epsilon, 0.0) ** 2))

    t_conv_norm = float(np.clip(t_conv_steps / max(t_post, 1), 0.0, 1.0))

    if c == 0:
        return large_penalty + v_self
    return v_self + t_conv_norm


def get_active_metric_builders() -> dict[str, EpisodeMetricFn]:
    """Return the benchmark metrics currently enabled."""
    return {
        "performance": compute_performance_metric,
        "efficiency": compute_efficiency_metric,
        "resilience": compute_resilience_metric,
        "adaptability": compute_adaptability_metric,
    }


def get_future_metric_todos() -> dict[str, str]:
    """Document where future benchmark metrics should be added."""
    return {}


def compute_episode_benchmark_metrics(
    episode_metrics: dict[str, list[Any]],
    metric_builders: dict[str, EpisodeMetricFn] | None = None,
) -> dict[str, float]:
    """Compute the active benchmark metrics for one episode."""
    builders = metric_builders or get_active_metric_builders()
    return {
        metric_name: metric_fn(episode_metrics)
        for metric_name, metric_fn in builders.items()
    }


def aggregate_benchmark_metrics(
    per_episode_metrics: dict[str, dict[str, float]],
    metric_builders: dict[str, EpisodeMetricFn] | None = None,
) -> dict[str, float]:
    """Aggregate active metrics across episodes using simple means."""
    builders = metric_builders or get_active_metric_builders()
    aggregate: dict[str, float] = {}

    for metric_name in builders:
        metric_values = [
            episode_metrics[metric_name]
            for episode_metrics in per_episode_metrics.values()
        ]
        aggregate[metric_name] = _mean_or_default(metric_values, default=0.0)

    aggregate["episodes_evaluated"] = len(per_episode_metrics)
    return aggregate


def compute_benchmark_summary(
    results_table: dict[str, dict[str, list[Any]]],
) -> dict[str, Any]:
    """Compute the currently enabled benchmark metrics from evaluation results.

    The benchmark tracks:
    - performance   : mean total reward per episode (higher is better)
    - efficiency    : mean decision time per action in seconds (lower is better)
    - resilience    : mean S_init * T_rec product across episodes (lower is better)
    - adaptability  : mean composite score encoding convergence, stability, and
      convergence time (lower is better)

    Both resilience and adaptability require episodes to supply a
    ``performance`` list and a ``disruption_step`` list.
    """

    builders = get_active_metric_builders()
    per_episode = {
        seed_key: compute_episode_benchmark_metrics(episode_metrics, builders)
        for seed_key, episode_metrics in results_table.items()
    }

    return {
        "metric_definitions": {
            "performance": "Mean total reward per episode. Higher is better.",
            "efficiency": "Mean decision time per action in seconds. Lower is better.",
            "resilience": (
                "Mean S_init * T_rec product per episode. "
                "S_init is the normalised depth of performance drop immediately "
                "after disruption; T_rec is the normalised steps to recovery. "
                "Lower is better."
            ),
            "adaptability": (
                "Composite score encoding convergence (C), stability (V_self), "
                "and convergence time (T_conv). Episodes that never converge "
                "receive a large penalty. Among converging episodes, lower "
                "V_self then lower T_conv wins. Lower is better."
            ),
        },
        "aggregate": aggregate_benchmark_metrics(per_episode, builders),
        "per_episode": per_episode,
        "todo_metrics": get_future_metric_todos(),
    }


def save_benchmark_summary(benchmark_summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(benchmark_summary, handle, indent=2)


def build_combined_benchmark_report(experiment_reports: list[dict[str, Any]]) -> dict[str, Any]:
    active_metric_names = tuple(get_active_metric_builders().keys())
    benchmark_rows = [report["benchmark"] for report in experiment_reports]
    episode_weights = [row["episodes_evaluated"] for row in benchmark_rows]

    aggregate = {
        metric_name: _weighted_mean_or_default(
            [row[metric_name] for row in benchmark_rows],
            episode_weights,
            default=0.0,
        )
        for metric_name in active_metric_names
    }
    aggregate["episodes_evaluated"] = int(
        sum(row["episodes_evaluated"] for row in benchmark_rows)
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "num_experiments": len(experiment_reports),
        "aggregate": aggregate,
        "experiments": experiment_reports,
        "todo_metrics": get_future_metric_todos(),
    }


def print_benchmark_summary(name_prefix: str, benchmark_summary: dict[str, Any]) -> None:
    aggregate = benchmark_summary["aggregate"]
    print(f"Benchmark Metrics:   {name_prefix}")
    print(f"  Performance:       {aggregate['performance']:.4f}")
    print(f"  Efficiency:        {aggregate['efficiency']:.6f}s")
    print(f"  Resilience:        {aggregate['resilience']:.6f}  (lower is better)")
    print(f"  Adaptability:      {aggregate['adaptability']:.6f}  (lower is better)")


def print_combined_benchmark_report(report: dict[str, Any]) -> None:
    print(f"\n{'='*50}")
    print("Overall Benchmark Summary")
    print(f"{'='*50}")
    print(f"Experiments:         {report['num_experiments']}")
    print(f"Performance:         {report['aggregate']['performance']:.4f}")
    print(f"Efficiency:          {report['aggregate']['efficiency']:.6f}s")
    print(f"Resilience:          {report['aggregate']['resilience']:.6f}  (lower is better)")
    print(f"Adaptability:        {report['aggregate']['adaptability']:.6f}  (lower is better)")
    print(f"Episodes Evaluated:  {report['aggregate']['episodes_evaluated']}")
    print(f"{'='*50}")
