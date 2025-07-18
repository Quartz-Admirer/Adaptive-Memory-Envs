import numpy as np
import random
from typing import Optional, List

class EndlessTMaze:
    """
    A standardized implementation of the T-Maze environment that supports multiple corridors.
    
    Action Space:
    - 0: Turn Left at junction
    - 1: Turn Right at junction
    - 2: Move Forward along the corridor
    - 3: Move Down / Backward (unused)
    
    State:
    - A numpy array [x, y, hint_value].
    - `x`: position along the corridor.
    - `y`: always 0.
    - `hint_value`: The hint (-1 or +1) is shown only at the start of the corridor (x=0).
                   It's 0 at all other positions.
    
    NOTE: Action id 3 corresponds to "go straight" while inside the corridor. At the T-junction
    (x = corridor_length) the agent must choose either action 0 (left) or 1 (right) depending on
    the hint. This mapping is consistent with the original RATE implementation.
    """
    def __init__(self, corridor_length: int = 10, num_corridors: int = 3, penalty: float = 0.0,
                 seed: Optional[int] = None, goal_reward: float = 1.0,
                 hints_override: Optional[List[int]] = None):
        self.corridor_length = corridor_length
        self.num_corridors = num_corridors
        self.penalty = penalty
        self.goal_reward = goal_reward
        self.max_steps = 6 * (corridor_length + 1) * num_corridors
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # If a fixed sequence of hints is provided, validate its length
        if hints_override is not None:
            assert len(hints_override) == num_corridors, (
                "hints_override length must match num_corridors"
            )
            self._fixed_hints = hints_override
        else:
            self._fixed_hints = None
        
        self.reset()
        #print(f"DEBUG: EndlessTMaze initialized with num_corridors={self.num_corridors}, goal_reward={self.goal_reward}, penalty={self.penalty}")
    
    def reset(self):
        self.current_corridor = 0
        self.x = 0
        self.y = 0
        self.done = False
        self.steps = 0
        
        # Hints are now -1 (go left) or +1 (go right). Allow override for testing.
        if self._fixed_hints is not None:
            self.hints = list(self._fixed_hints)  # copy to avoid mutation
        else:
            self.hints = [random.choice([-1, 1]) for _ in range(self.num_corridors)]
        self.current_hint = self.hints[0]
        
        return self.get_state()
    
    def step(self, action):
        # Accept numpy array or scalar; cast to int for internal logic
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        if self.done:
            return self.get_state(), 0, True, {}
        
        self.steps += 1
        reward = 0.0
        
        if self.x < self.corridor_length:
            # Agent is in the corridor
            if action == 2: # Move forward (action id 2 as in original RATE env)
                self.x += 1
            else:  # Wrong action in the corridor – penalize but keep episode running
                reward = self.penalty
                # Do not terminate: allow agent to correct itself and eventually
                # experience a positive reward at the junction. This greatly
                # reduces sparsity of positive signals during exploration.
        
        elif self.x == self.corridor_length:
            # Agent reached the T-junction and must turn.
            # Map hint (−1 / +1) to action id: −1 → 0 (turn left), +1 → 1 (turn right)
            correct_turn_action = 0 if self.current_hint == -1 else 1

            if action == correct_turn_action:
                reward = self.goal_reward
                self.current_corridor += 1

                if self.current_corridor < self.num_corridors:
                    # Move to the next corridor
                    self.x = 0
                    self.current_hint = self.hints[self.current_corridor]
                else:
                    # Finished all corridors successfully
                    self.done = True
            else:
                reward = self.penalty
                self.done = True
        
        if not self.done and self.steps >= self.max_steps:
            self.done = True  # no separate truncation flag in base env
        
        return self.get_state(), reward, self.done, {}
    
    def get_state(self):
        hint_to_show = self.current_hint if self.x == 0 else 0
        return np.array([self.x, self.y, hint_to_show], dtype=np.float32)

    def get_optimal_action(self):
        """Oracle policy: go forward until junction, then turn according to hint."""
        if self.x < self.corridor_length:
            return 2  # move forward inside the corridor
        else:
            # Map hint −1/1 to action 0/1 as above
            return 0 if self.current_hint == -1 else 1