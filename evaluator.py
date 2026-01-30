"""Competition evaluator. Runs the submitted agent against competition environments."""

import argparse
import gymnasium as gym
import AAMAS_Comp  # noqa: F401 -- triggers environment registration
from AAMAS_Comp.evaluation import run_complete_evaluation
from submission import get_agent


ENVIRONMENTS = {
    "ExampleNSFrozenLake-v0": "FrozenLake-v1",
    "ExampleNSCartPole-v0": "CartPole-v1",
    "ExampleNSAnt-v0": "Ant-v5",
}


def evaluate_local(num_episodes=10, start_seed=42):
    for env_id, base_env_id in ENVIRONMENTS.items():
        agent = get_agent(base_env_id)
        env = gym.make(
            env_id,
            change_notification=True,
            delta_change_notification=True,
            disable_env_checker=True,
            order_enforce=False,
        )

        run_complete_evaluation(
            env=env,
            agent=agent,
            start_seed=start_seed,
            num_episodes=num_episodes,
            name_prefix=env_id,
        )

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", choices=["local"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "local":
        evaluate_local(num_episodes=args.num_episodes, start_seed=args.start_seed)
