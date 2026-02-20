"""Submission file.

Configure and return an agent for each competition base environment.
"""

from pathlib import Path
from stable_baselines3 import PPO
from AAMAS_Comp.agent import MyModelBasedAgent, MyModelFreeAgent


def get_agent(env_id: str):
    """Return an agent instance configured for the given environment.

    Args:
        env_id: The base environment being evaluated. One of:
            - "FrozenLake-v1"
            - "CartPole-v1"
            - "Ant-v5"

    Returns:
        Agent: Your initialized agent object.
    """
    if env_id == "Ant-v5":
        model_path = Path("models/ppo_ant/ppo_ant.zip")
        model = PPO.load(str(model_path))
        return MyModelFreeAgent(model=model, deterministic=True)

    elif env_id == "FrozenLake-v1":
        # Minimal, valid model-based baseline.
        return MyModelBasedAgent(seed=42)

    elif env_id == "CartPole-v1":
        # Minimal, valid model-based baseline.
        return MyModelBasedAgent(seed=42)

    raise ValueError(f"{env_id} not in: Ant-v5, FrozenLake-v1, CartPole-v1")
