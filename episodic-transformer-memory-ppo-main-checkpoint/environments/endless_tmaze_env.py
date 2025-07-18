# endless_tmaze_env.py

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces
from typing import Optional, Tuple, Dict, Any, List

from environments.endless_tmaze import EndlessTMaze


class EndlessTMazeGym(Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        corridor_length: int = 11,
        num_corridors: int = 5,
        penalty: float = 0.0,
        goal_reward: float = 1.0,
        seed: Optional[int] = None,
        hints_override: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self._base_env = EndlessTMaze(
            corridor_length=corridor_length,
            num_corridors=num_corridors,
            penalty=penalty,
            goal_reward=goal_reward,
            seed=seed,
            hints_override=hints_override,
        )

        self.max_episode_steps = self._base_env.max_steps

        # Добавьте эти атрибуты для отслеживания статистики текущего эпизода
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._corridor_length = corridor_length # Можно использовать self._base_env.corridor_length
        self._norm = float(corridor_length) # Можно использовать float(self._base_env.corridor_length)

    def _state_to_obs(self, raw: np.ndarray) -> np.ndarray:
        x, _, hint = raw
        x_norm = x / self._norm
        hint_left = 1.0 if (x == 0 and hint == -1) else 0.0
        hint_right = 1.0 if (x == 0 and hint == 1) else 0.0
        return np.asarray([x_norm, hint_left, hint_right], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
        state = self._base_env.reset()
        
        # Сбрасываем статистику эпизода при каждом вызове reset
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        return self._state_to_obs(state), {}

    def step(self, action: int):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        # Логика маппинга действий (как у вас уже было)
        if action == 0:
            internal_action = 2 # Gym action 0 -> Base Move Forward
        elif action == 1:
            internal_action = 0 # Gym action 1 -> Base Turn Left
        elif action == 2:
            internal_action = 1 # Gym action 2 -> Base Turn Right
        else:
            # Обработка некорректного действия, если оно возможно
            # Для Gymnasium это обычно не предусмотрено, если action_space корректен
            # Но если Gym.action_space.sample() может вернуть что-то вне [0,1,2]
            # то стоит обработать или полагаться на base_env's penalty
            # Если base_env не обрабатывает 3, то это может быть:
            internal_action = 2 # По умолчанию двигаться вперед, если что-то пошло не так, или вызвать ошибку.
                                # В вашем случае base_env.step(3) ведет к penalty и done, что нормально.

        # Выполняем шаг в базовой среде
        obs_base, reward_base, done_base_conflated, info_base = self._base_env.step(internal_action)

        # Обновляем статистику текущего эпизода
        self.current_episode_reward += reward_base
        self.current_episode_length += 1

        # Определяем terminated и truncated согласно стандарту Gymnasium
        terminated = False
        truncated = False
        episode_info_to_return = {} # Словарь info для возврата в Gymnasium

        # Логика для разделения terminated и truncated
        # Базовая среда `endless_tmaze.py` устанавливает `self.done = True`
        # как для "естественного" завершения (успех/провал поворота), так и для "усечения" (достижение max_steps).
        # Мы должны корректно их разделить.

        # Признак усечения: достигнута максимальная длина эпизода, и при этом эпизод не был естественно завершен
        # Важно: `self._base_env.steps` уже инкрементирован в базовой среде.
        if self._base_env.steps >= self.max_episode_steps:
             # Если базовая среда завершилась из-за max_steps И это не был естественный финал
             # (т.е. агент не повернул правильно/неправильно и не закончил коридоры),
             # то это усечение.
             # Проверить "естественный" финал сложнее без явного флага из base_env.
             # Проще всего: если шаги достигли лимита, то это truncated, если нет другого явного признака terminated.
             # В PPO часто `truncated` просто означает, что эпизод закончился по TimeLimit.
             truncated = True
        
        # Признак завершения: базовая среда сигнализировала о завершении, и это не было усечением.
        # `done_base_conflated` может быть True из-за естественного завершения ИЛИ из-за `max_steps`.
        # Если `truncated` True, то `terminated` должен быть False (если только не было одновременно).
        if done_base_conflated and not truncated:
            # Это естественное завершение, если базовая среда сказала 'done' и это не было просто усечением.
            terminated = True
        elif done_base_conflated and truncated:
            # Если оба True, Gymnasium обычно отдает приоритет terminated, но для ясности лучше разделять.
            # В данном случае, если steps >= max_steps, то это truncation. Если еще и "естественно" закончилось,
            # то это более сложный случай. Для простоты, если достигли max_steps, считаем truncated.
            pass # Уже установлено truncated = True

        # Если эпизод завершился (либо terminated, либо truncated)
        if terminated or truncated:
            episode_info_to_return = {
                "reward": self.current_episode_reward,  # Общая награда за эпизод
                "length": self.current_episode_length,  # Общая длина эпизода
                # Можно добавить "success": True/False, если хотите отслеживать успех
                # Например: "success": self._base_env.current_corridor >= self._base_env.num_corridors
                # при условии, что это логически правильно для вашей задачи.
            }
            # Сбрасываем счетчики для следующего эпизода
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # Возвращаем кортеж, соответствующий стандарту Gymnasium
        return self._state_to_obs(obs_base), reward_base, terminated, truncated, episode_info_to_return