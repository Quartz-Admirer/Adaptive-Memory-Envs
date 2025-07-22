import numpy as np
import random
from typing import Optional, List, Dict

class EndlessTMaze:
    def __init__(self, length_config: Dict, num_corridors: int = 3, penalty: float = 0.0,
                 seed: Optional[int] = None, goal_reward: float = 1.0,
                 hints_override: Optional[List[int]] = None):

        self.length_mode = length_config.get("mode", "fixed")
        self.corridor_length_max = length_config.get("max", 10)
        self.corridor_length_min = length_config.get("min", 1)
        self.corridor_length = self.corridor_length_max

        self.num_corridors = num_corridors
        self.penalty = penalty
        self.goal_reward = goal_reward
        self.max_steps = 6 * (self.corridor_length_max + 1) * num_corridors
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._fixed_hints = hints_override
        self.reset()
    
    def reset(self):
        if self.length_mode == "uniform":
            self.corridor_length = random.randint(self.corridor_length_min, self.corridor_length_max)

        self.current_corridor = 0
        self.x = 0
        self.y = 0
        self.done = False
        self.steps = 0
        
        if self._fixed_hints is not None:
            self.hints = list(self._fixed_hints)
        else:
            self.hints = [random.choice([-1, 1]) for _ in range(self.num_corridors)]
        self.current_hint = self.hints[0]
        
        return self.get_state()
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        if self.done:
            return self.get_state(), 0, True, {}
        
        self.steps += 1
        reward = 0.0
        
        if self.x < self.corridor_length:
            if action == 2:
                self.x += 1
            else:
                reward = self.penalty
        
        elif self.x == self.corridor_length:
            correct_turn_action = 0 if self.current_hint == -1 else 1

            if action == correct_turn_action:
                reward = self.goal_reward
                self.current_corridor += 1

                if self.current_corridor < self.num_corridors:
                    self.x = 0
                    self.current_hint = self.hints[self.current_corridor]
                    if self.length_mode == "uniform":
                         self.corridor_length = random.randint(self.corridor_length_min, self.corridor_length_max)
                else:
                    self.done = True
            else:
                reward = self.penalty
                self.done = True
        
        if not self.done and self.steps >= self.max_steps:
            self.done = True
        
        return self.get_state(), reward, self.done, {}
    
    def get_state(self):
        hint_to_show = self.current_hint if self.x == 0 else 0
        return np.array([self.x, self.y, hint_to_show], dtype=np.float32)

