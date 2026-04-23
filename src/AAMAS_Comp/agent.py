from copy import deepcopy
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ns_gym.benchmark_algorithms import MCTS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent


class _DiscreteActionPlanningEnv:
    """Expose a discrete planning interface over a continuous-action env."""

    def __init__(self, env, action_candidates):
        self._env = env
        self._action_candidates = np.asarray(
            action_candidates, dtype=env.action_space.dtype
        )
        self.action_space = spaces.Discrete(len(self._action_candidates))
        self.observation_space = env.observation_space

    def step(self, action_idx):
        action = self.decode_action(action_idx)
        return self._env.step(action)

    def decode_action(self, action_idx):
        return self._action_candidates[int(action_idx)].copy()

    def __deepcopy__(self, memo):
        copied_env = deepcopy(self._env, memo)
        copied_candidates = self._action_candidates.copy()
        return _DiscreteActionPlanningEnv(copied_env, copied_candidates)

    def __getattr__(self, name):
        return getattr(self._env, name)


class _SafeMCTS(MCTS):
    """MCTS variant that tolerates partially explored action sets."""

    def search(self):
        for _ in range(self.m):
            self.sim_env = deepcopy(self.env)
            leaf_node = self._tree_policy(self.v0)
            expanded_node = self._expand(leaf_node)
            if hasattr(expanded_node, "action"):
                expanded_node = self._expand(expanded_node)
            rollout_return = self._default_policy(expanded_node)
            self._backpropagation(rollout_return, expanded_node)

        action_values = [
            self.Qsa.get((self.v0.state, action), float("-inf"))
            for action in self.possible_actions
        ]
        if all(np.isneginf(value) for value in action_values):
            best_action = int(np.random.choice(self.possible_actions))
        else:
            best_action = int(np.argmax(action_values))
        return best_action, action_values


class MyModelBasedAgent(ModelBasedAgent):

    def __init__(self, d, m, c=1.4, gamma=0.99, continuous_action_samples=16) -> None:
        super().__init__()

        self.d = d
        self.m = m
        self.c = c
        self.gamma = gamma
        self.continuous_action_samples = continuous_action_samples

    def _build_continuous_action_candidates(self, action_space):
        low = np.where(np.isfinite(action_space.low), action_space.low, -1.0)
        high = np.where(np.isfinite(action_space.high), action_space.high, 1.0)
        center = np.clip(np.zeros_like(low), low, high).astype(action_space.dtype)

        candidates = [center]

        for idx in range(action_space.shape[0]):
            positive = center.copy()
            negative = center.copy()
            positive[idx] = high[idx]
            negative[idx] = low[idx]
            candidates.append(positive)
            candidates.append(negative)

        if self.continuous_action_samples > 0:
            rng = np.random.default_rng(0)
            random_samples = rng.uniform(
                low=low,
                high=high,
                size=(self.continuous_action_samples, action_space.shape[0]),
            )
            candidates.extend(random_samples.astype(action_space.dtype))

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            clipped = np.clip(candidate, low, high).astype(
                action_space.dtype, copy=False
            )
            key = tuple(np.asarray(clipped, dtype=np.float32).round(6))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(clipped.copy())

        return np.asarray(unique_candidates, dtype=action_space.dtype)

    def get_action(self, obs: Dict, planning_env):
        state = obs["state"]

        search_env = planning_env
        decode_action = None

        if isinstance(planning_env.action_space, spaces.Box):
            action_candidates = self._build_continuous_action_candidates(
                planning_env.action_space
            )
            search_env = _DiscreteActionPlanningEnv(planning_env, action_candidates)
            decode_action = search_env.decode_action

        mcts_solver = _SafeMCTS(
            env=search_env,
            state=state,
            d=self.d,
            m=self.m,
            c=self.c,
            gamma=self.gamma,
        )

        action, _ = mcts_solver.search()

        if decode_action is not None:
            return decode_action(action)

        return action

    # def set_seed(self, seed):
    #     self._rng = np.random.default_rng(seed)

class MyModelFreeAgent(ModelFreeAgent):
    """Minimal model-free agent with optional lazy-loaded SB3 PPO inference."""

    def __init__(
        self,
        model=None,
        action_space=None,
        deterministic=True,
        vec_normalize=None,
        model_path=None,
        vec_normalize_path=None,
        base_env_id=None,
        policy="MlpPolicy",
        ppo_kwargs=None,
        allow_fresh_ppo=False,
        seed=None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self.model = model
        self.action_space = action_space
        self.deterministic = deterministic
        self.vec_normalize = vec_normalize
        self.model_path = Path(model_path) if model_path is not None else None
        self.vec_normalize_path = (
            Path(vec_normalize_path) if vec_normalize_path is not None else None
        )
        self.base_env_id = base_env_id
        self.policy = policy
        self.ppo_kwargs = dict(ppo_kwargs or {})
        self.allow_fresh_ppo = allow_fresh_ppo
        self._vec_env = None

    @classmethod
    def from_ppo(
        cls,
        base_env_id,
        model_path=None,
        vec_normalize_path=None,
        deterministic=True,
        policy="MlpPolicy",
        ppo_kwargs=None,
        allow_fresh_ppo=False,
    ):
        return cls(
            model_path=model_path,
            vec_normalize_path=vec_normalize_path,
            base_env_id=base_env_id,
            deterministic=deterministic,
            policy=policy,
            ppo_kwargs=ppo_kwargs,
            allow_fresh_ppo=allow_fresh_ppo,
        )

    def _ensure_ppo_model_loaded(self):
        if self.model is not None:
            return

        if self.model_path is not None and self.model_path.exists():
            self.model = PPO.load(str(self.model_path))
        else:
            if self.model_path is not None and not self.allow_fresh_ppo:
                raise FileNotFoundError(f"PPO model not found at {self.model_path}")
            if self.base_env_id is None:
                return

            env = gym.make(self.base_env_id)
            self.model = PPO(
                self.policy,
                env,
                verbose=0,
                seed=int(self._rng.integers(0, 2**31 - 1)),
                **self.ppo_kwargs,
            )

        if self.vec_normalize_path is None:
            return

        if not self.vec_normalize_path.exists():
            raise FileNotFoundError(
                f"VecNormalize stats not found at {self.vec_normalize_path}"
            )
        if self.base_env_id is None:
            raise ValueError("base_env_id is required when loading VecNormalize stats.")

        self._vec_env = DummyVecEnv([lambda: gym.make(self.base_env_id)])
        self.vec_normalize = VecNormalize.load(
            str(self.vec_normalize_path),
            self._vec_env,
        )
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False

    def get_action(self, obs):
        self._ensure_ppo_model_loaded()

        if self.model is not None:
            state = obs["state"]
            if self.vec_normalize is not None:
                state = self.vec_normalize.normalize_obs(state)
            action, _ = self.model.predict(state, deterministic=self.deterministic)
            if isinstance(action, np.ndarray) and action.shape == ():
                return action.item()
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
