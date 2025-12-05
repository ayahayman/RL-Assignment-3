"""
Wrappers to convert discrete environments to continuous action spaces for SAC.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousCartPole(gym.Wrapper):
    """Wrapper to make CartPole-v1 use continuous actions."""
    
    def __init__(self, env):
        super().__init__(env)
        # Continuous action space: force magnitude in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        # Convert continuous action to discrete
        # action is in [-1, 1], map to {0, 1}
        # Use threshold at 0 for balanced control
        # 0 = push left, 1 = push right
        discrete_action = 1 if action[0] >= 0 else 0
        return self.env.step(discrete_action)


class ContinuousMountainCar(gym.Wrapper):
    """Wrapper to make MountainCar-v0 use continuous actions."""
    
    def __init__(self, env):
        super().__init__(env)
        # Continuous action space: force in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        # Convert continuous action to discrete
        # action in [-1, 1] maps to {0, 1, 2} (left, no push, right)
        if action[0] < -0.33:
            discrete_action = 0  # push left
        elif action[0] > 0.33:
            discrete_action = 2  # push right
        else:
            discrete_action = 1  # no push
        return self.env.step(discrete_action)


class ContinuousAcrobot(gym.Wrapper):
    """Wrapper to make Acrobot-v1 use continuous actions."""
    
    def __init__(self, env):
        super().__init__(env)
        # Continuous action space: torque in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        # Convert continuous action to discrete
        # action in [-1, 1] maps to {0, 1, 2} (negative, zero, positive torque)
        if action[0] < -0.33:
            discrete_action = 0  # negative torque
        elif action[0] > 0.33:
            discrete_action = 2  # positive torque
        else:
            discrete_action = 1  # zero torque
        return self.env.step(discrete_action)
