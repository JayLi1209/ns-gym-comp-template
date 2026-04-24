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


def _get_normalized_performance(episode_metrics: dict[str, list[Any]]) -> np.ndarray:
    """Return the normalized per-step performance series if present."""
    return np.asarray(episode_metrics.get("performance", []), dtype=float)


def _get_disruption_step(episode_metrics: dict[str, list[Any]]) -> int | None:
    """Return the first disruption step, using fallbacks when needed."""
    disruption_steps = episode_metrics.get("disruption_step", [])
    if disruption_steps:
        return int(disruption_steps[0])

    env_change = episode_metrics.get("env_change", [])
    for i, changed in enumerate(env_change):
        if bool(changed):
            return i

    return None


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


def _compute_resilience_components(
    episode_metrics: dict[str, list[Any]],
    epsilon: float = 0.9,
    window: int = 5,
) -> dict[str, float]:
    """Compute the resilience component metrics for one episode."""
    perf = _get_normalized_performance(episode_metrics)
    t_d = _get_disruption_step(episode_metrics)
    window = max(int(window), 1)

    if perf.size == 0 or t_d is None or t_d < 0 or t_d >= len(perf):
        return {
            "resilience_s_init": 1.0,
            "resilience_t_rec": 1.0,
        }

    p_at_disruption = float(perf[t_d])
    s_init = float(np.clip(1.0 - p_at_disruption, 0.0, 1.0))

    recovery_threshold = float(epsilon)
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

    return {
        "resilience_s_init": s_init,
        "resilience_t_rec": t_rec_norm,
    }


def compute_resilience_metric(
    episode_metrics: dict[str, list[Any]],
    epsilon: float = 0.9,
    window: int = 5,
) -> float:
    """Compute the resilience score S_init * T_rec for one episode.

    Resilience measures how badly the agent stumbled immediately after a
    disruption (S_init) and how long it took to recover (T_rec).  A lower
    product is better — an ideal agent absorbs the disruption quickly and with
    little performance loss.

    The metric follows the (S_init, T_rec) formulation from the slides
    (slides 9, 19–20) and the infrastructure-resilience survey
    (https://arxiv.org/pdf/2102.01009).

    Expected keys in episode_metrics
    ---------------------------------
    performance : list[float]
        Per-timestep normalized performance signal p(t).
    disruption_step : list[int]  (length 1)
        Index into `performance` at which the disruption occurred (t_d).
        If absent, the first non-zero entry in ``env_change`` is used.

    Parameters
    ----------
    epsilon:
        Recovery threshold applied directly to normalized performance.
    window:
        Number of consecutive post-disruption steps required to count as
        recovered.

    Returns
    -------
    float
        S_init * T_rec, normalised so that both factors lie in [0, 1] when
        possible. Returns 1.0 (worst) when no usable disruption/performance
        data is recorded or the agent never recovers within the episode.
    """
    components = _compute_resilience_components(
        episode_metrics,
        epsilon=epsilon,
        window=window,
    )
    return float(
        components["resilience_s_init"] * components["resilience_t_rec"]
    )


def compute_resilience_s_init_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return the resilience depth-of-impact component S_init."""
    return float(_compute_resilience_components(episode_metrics)["resilience_s_init"])


def compute_resilience_t_rec_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return the resilience recovery-time component T_rec."""
    return float(_compute_resilience_components(episode_metrics)["resilience_t_rec"])


def _compute_adaptability_components(
    episode_metrics: dict[str, list[Any]],
    eta_star_fraction: float = 0.8,
    convergence_window: int = 10,
    convergence_beta: float = 0.9,
    stability_window: int = 5,
    stability_epsilon: float = 0.05,
    stability_xi: float = 1e-6,
) -> dict[str, float]:
    """Compute the adaptability component metrics for one episode."""
    perf = _get_normalized_performance(episode_metrics)
    t_d = _get_disruption_step(episode_metrics)
    convergence_window = max(int(convergence_window), 1)
    stability_window = max(int(stability_window), 1)

    if perf.size == 0 or t_d is None or t_d < 0 or t_d >= len(perf):
        return {
            "adaptability_c": 0.0,
            "adaptability_v_self": 1.0,
            "adaptability_t_conv": 1.0,
        }

    post_perf = perf[t_d:]
    t_post = len(post_perf)
    eta_star = float(eta_star_fraction)

    converged = False
    t_conv_steps = t_post

    for i in range(t_post - convergence_window + 1):
        window_slice = post_perf[i : i + convergence_window]
        fraction_above = float(np.mean(window_slice >= eta_star))
        if fraction_above >= convergence_beta:
            converged = True
            t_conv_steps = i
            break

    c = 1.0 if converged else 0.0

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

    return {
        "adaptability_c": c,
        "adaptability_v_self": v_self,
        "adaptability_t_conv": t_conv_norm,
    }


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

    Adaptability is judged by three criteria (slide 3):
      1. Convergence  C  — does the agent eventually reach a desirable threshold?
      2. Stability V_self — how much does performance oscillate post-shift?
      3. Convergence time T_conv — how quickly does it reach the threshold?

    Ranking priority (slide 16): C=1 > C=0; then lower V_self; then lower T_conv.

    This function encodes that priority into a single float so that standard
    numerical ranking (lower = better) reproduces the lexicographic order:

        score = (1 - C) * LARGE_PENALTY
                + C * (V_self + T_conv_norm)

    where LARGE_PENALTY ensures any non-converging episode ranks worse than any
    converging one, regardless of its V_self or T_conv.

    Expected keys in episode_metrics
    ---------------------------------
    performance : list[float]
        Per-timestep normalized performance signal p(t).
    disruption_step : list[int]  (length 1)
        Index into `performance` at which the environment shift occurred (t_d).
        If absent, the first non-zero entry in ``env_change`` is used.

    Parameters
    ----------
    eta_star_fraction:
        Desired normalized performance threshold eta*.
    convergence_window:
        Delta for the convergence check C^(Delta, beta) (slide 12).
    convergence_beta:
        Fraction of the convergence window that must exceed eta* (slide 12).
    stability_window:
        Delta for the local reference performance p̄_Delta (slide 13).
    stability_epsilon:
        Tolerance for fluctuations in V_self (slide 14).
    stability_xi:
        Small constant xi to avoid division by zero in d_Delta (slide 14).

    Returns
    -------
    float
        Lower is better.  Returns LARGE_PENALTY (1e6) when no disruption data
        is available.
    """
    LARGE_PENALTY = 1e6

    components = _compute_adaptability_components(
        episode_metrics,
        eta_star_fraction=eta_star_fraction,
        convergence_window=convergence_window,
        convergence_beta=convergence_beta,
        stability_window=stability_window,
        stability_epsilon=stability_epsilon,
        stability_xi=stability_xi,
    )
    c = components["adaptability_c"]
    v_self = components["adaptability_v_self"]
    t_conv = components["adaptability_t_conv"]

    if c == 0.0:
        return LARGE_PENALTY + v_self
    return v_self + t_conv


def compute_adaptability_c_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return the adaptability convergence indicator C."""
    return float(_compute_adaptability_components(episode_metrics)["adaptability_c"])


def compute_adaptability_v_self_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return the adaptability stability metric V_self."""
    return float(
        _compute_adaptability_components(episode_metrics)["adaptability_v_self"]
    )


def compute_adaptability_t_conv_metric(episode_metrics: dict[str, list[Any]]) -> float:
    """Return the adaptability convergence-time metric T_conv."""
    return float(
        _compute_adaptability_components(episode_metrics)["adaptability_t_conv"]
    )


def get_active_metric_builders() -> dict[str, EpisodeMetricFn]:
    """Return the benchmark metrics currently enabled."""
    return {
        "performance": compute_performance_metric,
        "efficiency": compute_efficiency_metric,
        "resilience_s_init": compute_resilience_s_init_metric,
        "resilience_t_rec": compute_resilience_t_rec_metric,
        "resilience": compute_resilience_metric,
        "adaptability_c": compute_adaptability_c_metric,
        "adaptability_v_self": compute_adaptability_v_self_metric,
        "adaptability_t_conv": compute_adaptability_t_conv_metric,
        "adaptability": compute_adaptability_metric,
    }


def get_future_metric_todos() -> dict[str, str]:
    """Document where future benchmark metrics should be added."""
    # resilience and adaptability are now implemented — nothing left to do.
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
    - resilience_s_init : mean initial disruption impact (lower is better)
    - resilience_t_rec  : mean normalised recovery time (lower is better)
    - resilience    : mean S_init * T_rec product across episodes (lower is better)
    - adaptability_c      : mean convergence indicator C (higher is better)
    - adaptability_v_self : mean stability score V_self (lower is better)
    - adaptability_t_conv : mean normalised convergence time (lower is better)
    - adaptability  : mean composite score encoding convergence, stability, and
                      convergence time (lower is better)

    Both resilience and adaptability require episodes to supply a
    ``performance`` list (per-step p(t)) and either a ``disruption_step`` list
    or an ``env_change`` trace from which the first disruption can be inferred.
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
            "resilience_s_init": (
                "Mean initial disruption impact S_init per episode. Lower is better."
            ),
            "resilience_t_rec": (
                "Mean normalised recovery time T_rec per episode. Lower is better."
            ),
            "resilience": (
                "Mean S_init * T_rec product per episode. "
                "S_init is the normalised depth of performance drop immediately "
                "after disruption; T_rec is the normalised steps to recovery. "
                "Lower is better."
            ),
            "adaptability_c": (
                "Mean convergence indicator C per episode. Higher is better."
            ),
            "adaptability_v_self": (
                "Mean stability score V_self per episode. Lower is better."
            ),
            "adaptability_t_conv": (
                "Mean normalised convergence time T_conv per episode. Lower is better."
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
    print(f"  Resilience S_init: {aggregate['resilience_s_init']:.6f}  (lower is better)")
    print(f"  Resilience T_rec:  {aggregate['resilience_t_rec']:.6f}  (lower is better)")
    print(f"  Resilience Score:  {aggregate['resilience']:.6f}  (lower is better)")
    print(f"  Adaptability C:    {aggregate['adaptability_c']:.6f}  (higher is better)")
    print(f"  Adaptability V_self:{aggregate['adaptability_v_self']:.6f}  (lower is better)")
    print(f"  Adaptability T_conv:{aggregate['adaptability_t_conv']:.6f}  (lower is better)")
    print(f"  Adaptability Score: {aggregate['adaptability']:.6f}  (lower is better)")


def print_combined_benchmark_report(report: dict[str, Any]) -> None:
    print(f"\n{'='*50}")
    print("Overall Benchmark Summary")
    print(f"{'='*50}")
    print(f"Experiments:         {report['num_experiments']}")
    print(f"Performance:         {report['aggregate']['performance']:.4f}")
    print(f"Efficiency:          {report['aggregate']['efficiency']:.6f}s")
    print(f"Resilience S_init:   {report['aggregate']['resilience_s_init']:.6f}  (lower is better)")
    print(f"Resilience T_rec:    {report['aggregate']['resilience_t_rec']:.6f}  (lower is better)")
    print(f"Resilience Score:    {report['aggregate']['resilience']:.6f}  (lower is better)")
    print(f"Adaptability C:      {report['aggregate']['adaptability_c']:.6f}  (higher is better)")
    print(f"Adaptability V_self: {report['aggregate']['adaptability_v_self']:.6f}  (lower is better)")
    print(f"Adaptability T_conv: {report['aggregate']['adaptability_t_conv']:.6f}  (lower is better)")
    print(f"Adaptability Score:  {report['aggregate']['adaptability']:.6f}  (lower is better)")
    print(f"Episodes Evaluated:  {report['aggregate']['episodes_evaluated']}")
    print(f"{'='*50}")
