from time import perf_counter


def run_single_episode(env, agent, seed):

    obs, reward = env.reset(seed=seed)
    agent.set_seed(seed=seed)

    done = False
    truncated = False
    
    episode_metrics = {
        "rewards": [],
        "observations": [],
        "actions": [],
        "decision_time": []
    }

    while not (done or truncated):

        action, decision_time = agent.validate_and_get_action(obs)

        obs, reward, done, truncated, info = env.step(action)
        
        episode_metrics["observations"] = obs
        episode_metrics["rewards"] = reward.reward
        episode_metrics["actions"].append(action)
        episode_metrics["decision_time"].append(decision_time)

    return episode_metrics


def run_complete_evaluation(env, agent, start_seed):

    raise NotImplementedError






    














