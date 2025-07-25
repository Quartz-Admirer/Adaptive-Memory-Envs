from __future__ import annotations
import numpy as np
from gymnasium import Env, spaces
from typing import Optional, Tuple, Dict, Any

from popgym.envs.endless_tmaze import EndlessTMaze

class EndlessTMazeGym(Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any], seed: Optional[int] = None) -> None:
        super().__init__()
        
        # Адаптивная обработка конфига для совместимости
        if "lengths" in env_config:
            length_config = env_config["lengths"]
            num_corridors = env_config.get("num_corridors", 5)
            penalty = env_config.get("penalty", -0.01)
            goal_reward = env_config.get("goal_reward", 1.0)
        else:
            print("Warning: 'lengths' key not found. Using old config structure.")
            corridor_length = env_config.get("corridor_length", 10)
            length_config = {"mode": "fixed", "max": corridor_length}
            num_corridors = env_config.get("num_corridors", 5)
            penalty = env_config.get("penalty", -0.01)
            goal_reward = env_config.get("goal_reward", 1.0)

        self._base_env = EndlessTMaze(
            length_config=length_config,
            num_corridors=num_corridors,
            penalty=penalty,
            goal_reward=goal_reward,
            seed=seed
        )

        self.max_episode_steps = self._base_env.max_steps
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        self._norm = float(self._base_env.corridor_length_max)

    def _state_to_obs(self, raw: np.ndarray) -> np.ndarray:
        x, _, hint = raw
        x_norm = x / self._norm if self._norm > 0 else 0.0
        hint_left = 1.0 if (x == 0 and hint == -1) else 0.0
        hint_right = 1.0 if (x == 0 and hint == 1) else 0.0
        return np.asarray([x_norm, hint_left, hint_right], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
        state = self._base_env.reset()
        
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        return self._state_to_obs(state), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        mapping = {0: 2, 1: 0, 2: 1}
        internal_action = mapping.get(action, 2)

        obs_base, reward_base, done_base, info_base = self._base_env.step(internal_action)

        self.current_episode_reward += reward_base
        self.current_episode_length += 1

        terminated = False
        truncated = False
        episode_info = {}

        if done_base:
            if self._base_env.steps >= self.max_episode_steps:
                truncated = True
            else:
                terminated = True
            
            episode_info = {
                "reward": self.current_episode_reward,
                "length": self.current_episode_length,
                "success": 1.0 if self._base_env.current_corridor >= self._base_env.num_corridors else 0.0
            }

        return self._state_to_obs(obs_base), reward_base, terminated, truncated, episode_info

