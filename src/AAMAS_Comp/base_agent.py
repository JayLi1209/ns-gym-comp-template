from ns_gym.base import Agent
import gymnasium as gym
from typing import Dict
import numpy as np 
import time

"""Make sure your agents inherit from these classes so that they adhere to the required interfaces. 

Don't change this code as we have our own versions for evaluation. 
"""

class ModelBasedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, obs: Dict, planning_env: gym.Env):
        raise NotImplementedError
    
    def act(self, obs, planning_env):
        """Agent sub-class requries this method to be implemented
        """
        return self.get_action(obs, planning_env)
    
    def validate_and_get_action(self, obs: Dict, planning_env: gym.Env):
        """Called by the competition evaluator."""
        start_time = time.perf_counter()
        action = self.get_action(obs, planning_env)
        end_time = time.perf_counter()
        if not isinstance(action, (np.ndarray, int, np.integer)):
            raise TypeError(f"Action must be a numpy array or int, got {type(action)}")

        if not planning_env.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the bounds of {planning_env.action_space}")
        

        decision_time = end_time - start_time
    
        return action, decision_time


class ModelFreeAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, obs: Dict):
        raise NotImplementedError

    def act(self, obs):
        return self.get_action(obs)
    
    def validate_and_get_action(self, obs: Dict, action_space: gym.Space):
        """Called by the competition evaluator."""
        start_time = time.perf_counter()
        action = self.get_action(obs)
        end_time = time.perf_counter()

        assert action_space.contains(action), f"Invalid action: {action}"

        decision_time = end_time - start_time

        return action, decision_time
