"""Submission file.

Configure and return an agent for each competition base environment.
"""

from pathlib import Path

from AAMAS_Comp.agent import MyModelFreeAgent
from AAMAS_Comp.agent import MyModelBasedAgent


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
        return MyModelBasedAgent(d=50, m=100, c=1.4, gamma=0.99)

    elif env_id == "FrozenLake-v1":
        return MyModelBasedAgent(d=50, m=100, c=1.4, gamma=0.99)

    elif env_id == "CartPole-v1":
        return MyModelBasedAgent(d=50, m=100, c=1.4, gamma=0.99)

    raise ValueError(f"{env_id} not in: Ant-v5, FrozenLake-v1, CartPole-v1")
