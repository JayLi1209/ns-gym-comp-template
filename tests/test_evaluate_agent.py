from AAMAS_Comp.evaluation.evaluate_agent import run_single_episode


def test_run_single_episode_populates_frozenlake_performance(
    ns_frozenlake_env,
    dummy_model_based_agent,
):
    episode = run_single_episode(ns_frozenlake_env, dummy_model_based_agent, seed=0)

    assert len(episode["performance"]) == len(episode["rewards"])
    assert episode["disruption_step"]
    assert all(0.0 <= value <= 1.0 for value in episode["performance"])


def test_run_single_episode_populates_cartpole_performance(
    ns_cartpole_env,
    dummy_model_free_agent_factory,
):
    agent = dummy_model_free_agent_factory(ns_cartpole_env.action_space)
    episode = run_single_episode(ns_cartpole_env, agent, seed=0)

    assert len(episode["performance"]) == len(episode["rewards"])
    assert episode["disruption_step"]
    assert all(0.0 <= value <= 1.0 for value in episode["performance"])
