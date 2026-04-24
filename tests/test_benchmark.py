import pytest

from AAMAS_Comp.evaluation.benchmark import compute_benchmark_summary


def test_benchmark_summary_reports_resilience_and_adaptability_components():
    performance = [1.0, 0.2, 0.1] + [0.95] * 13
    results_table = {
        "42": {
            "rewards": [1.0, 0.0],
            "decision_time": [0.1, 0.2],
            "performance": performance,
            "env_change": [0, 1] + [1] * (len(performance) - 2),
        }
    }

    summary = compute_benchmark_summary(results_table)
    per_episode = summary["per_episode"]["42"]
    aggregate = summary["aggregate"]

    assert per_episode["resilience_s_init"] == pytest.approx(0.8)
    assert per_episode["resilience_t_rec"] == pytest.approx(2 / 15)
    assert per_episode["resilience"] == pytest.approx(
        per_episode["resilience_s_init"] * per_episode["resilience_t_rec"]
    )

    assert per_episode["adaptability_c"] == pytest.approx(1.0)
    assert per_episode["adaptability_t_conv"] == pytest.approx(1 / 15)
    assert per_episode["adaptability"] == pytest.approx(
        per_episode["adaptability_v_self"] + per_episode["adaptability_t_conv"]
    )

    assert aggregate["resilience_s_init"] == per_episode["resilience_s_init"]
    assert aggregate["resilience_t_rec"] == per_episode["resilience_t_rec"]
    assert aggregate["adaptability_c"] == per_episode["adaptability_c"]
    assert aggregate["adaptability_t_conv"] == per_episode["adaptability_t_conv"]
