from typing import Dict
import numpy as np
from gymnasium import spaces
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent


class MyModelBasedAgent(ModelBasedAgent):
    """Random actions!"""

    def __init__(self, seed=None):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def get_action(self, obs: Dict, planning_env):
        del obs  # ignore observations...
        action_space = planning_env.action_space

        # For FrozenLake and CartPole
        if isinstance(action_space, spaces.Discrete):
            return int(self._rng.integers(action_space.n))

    # def set_seed(self, seed):
    #     self._rng = np.random.default_rng(seed)

class MyModelFreeAgent(ModelFreeAgent):
    """Minimal model-free agent with optional SB3 model inference."""

    def __init__(self, model=None, action_space=None, deterministic=True, vec_normalize=None, seed=None):
        super().__init__()
        self.model = model
        self.action_space = action_space
        self.deterministic = deterministic
        self.vec_normalize = vec_normalize

    def get_action(self, obs):
        # Only implemented for Ant-v5
        if self.model is not None:
            state = obs["state"]
            if self.vec_normalize is not None:
                state = self.vec_normalize.normalize_obs(state)
            action, _ = self.model.predict(state, deterministic=self.deterministic)
            return action

        if self.action_space is None:
            raise ValueError("Provide either a trained model or an action_space.")

        if isinstance(self.action_space, spaces.Box):
            low = np.where(np.isfinite(self.action_space.low), self.action_space.low, -1.0)
            high = np.where(np.isfinite(self.action_space.high), self.action_space.high, 1.0)
            return self._rng.uniform(low, high).astype(self.action_space.dtype)

        return self.action_space.sample()

    # def set_seed(self, seed):
    #     self._rng = np.random.default_rng(seed)
    #     if self.action_space is not None and hasattr(self.action_space, "seed"):
    #         self.action_space.seed(seed)
