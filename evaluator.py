"""Competition evaluator. Runs the submitted agent against competition environments."""

import argparse
import json
import sys
from pathlib import Path

# Allow running `python evaluator.py` without installing the package first.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
import AAMAS_Comp  # noqa: F401 -- triggers environment registration
from AAMAS_Comp.evaluation import (
    build_combined_benchmark_report,
    print_combined_benchmark_report,
    run_complete_evaluation,
)

from collections import namedtuple

from submission import get_agent

#Uncomment this line to run baseline models from example submission. 
# from example_submission import get_agent

#Add additional envionments here or comment out the ones you do not want to evaluate. 
ENVIRONMENTS = {
    "ExampleNSFrozenLake-v0": "FrozenLake-v1",
    "ExampleNSCartPole-v0": "CartPole-v1",
    "ExampleNSAnt-v0": "Ant-v5",
}



NotificationSetting = namedtuple("Notification",["change_notification", "delta_change_notification", "label"])

# This sets the notifiation levels. Comment out the levels you do not want to evaluate
NOTIFICATIONS = [NotificationSetting(True,  True,  "notify-full"),
                 NotificationSetting(True,  False, "notify-change"),
                 NotificationSetting(False, False, "notify-none")]



def evaluate_local(num_episodes=10, start_seed=42):
    benchmark_runs = []

    for env_id, base_env_id in ENVIRONMENTS.items():
        for change_notification, delta_change_notification, notify_label in NOTIFICATIONS:

            agent = get_agent(base_env_id)
            env = gym.make(
                env_id,
                change_notification=change_notification,
                delta_change_notification=delta_change_notification,
                disable_env_checker=True,
                order_enforce=False,
            )

            name_prefix = f"{env_id}__{notify_label}"

            evaluation_artifacts = run_complete_evaluation(
                env=env,
                agent=agent,
                start_seed=start_seed,
                num_episodes=num_episodes,
                name_prefix=name_prefix,
                return_artifacts=True,
            )

            benchmark_runs.append(
                {
                    "name_prefix": name_prefix,
                    "environment_id": env_id,
                    "base_environment_id": base_env_id,
                    "notification_label": notify_label,
                    "change_notification": change_notification,
                    "delta_change_notification": delta_change_notification,
                    "experiment_dir": str(evaluation_artifacts["experiment_dir"]),
                    "benchmark": evaluation_artifacts["benchmark"]["aggregate"],
                }
            )

            env.close()

    results_dir = PROJECT_ROOT / "results"
    combined_report = build_combined_benchmark_report(benchmark_runs)
    benchmark_path = results_dir / "benchmark_summary.json"
    results_dir.mkdir(exist_ok=True)

    with benchmark_path.open("w") as handle:
        json.dump(combined_report, handle, indent=2)

    print_combined_benchmark_report(combined_report)
    print(f"Combined benchmark file: {benchmark_path}")

    return combined_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", choices=["local"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "local":
        evaluate_local(num_episodes=args.num_episodes, start_seed=args.start_seed)
