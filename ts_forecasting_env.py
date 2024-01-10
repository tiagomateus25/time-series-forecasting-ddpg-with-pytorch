#!/usr/bin/env python3
import numpy as np
from gym import Env
from gym.spaces import Box
from typing import Optional


class ts_forecasting_env(Env):
    def __init__(self, historical_dp=10, trajectory=None, data=None, render_mode: Optional[str] = None):
        # Data
        self.data = data

        # Trajectory
        self.trajectory = trajectory
        
        # Number of historical data points
        self.historical_dp = historical_dp

        # Low states
        low = np.zeros([self.historical_dp], dtype=np.float32)
        
        # High states
        high = np.ones([self.historical_dp], dtype=np.float32)

        # Define the action and state spaces
        self.action_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low, high, shape=(self.historical_dp,), dtype=np.float32)

        # Empty array to store chosen actions for the last episode
        self.actions = np.array([])

        # Render
        self.render_mode = render_mode

    def reset(self):
        # Reset the environment to an initial state
        self.iteration = 0

        # Random initial state
        self.index = np.random.choice(range(self.historical_dp, len(self.data)))
        self.state = np.array(self.data[self.index - self.historical_dp:self.index], dtype=np.float32)

        return np.array(self.state, dtype=np.float32)
    
    def step(self, action):
        # Get the current state
        self.current_state = self.state

        # Calculate the reward
        reward = - abs(self.data[self.index + self.iteration] - action)

        # Store chosen action
        self.chosen_action = action

        # Calculate the next state
        self.iteration += 1
        self.state = np.array(self.data[self.index - self.historical_dp + self.iteration:self.index + self.iteration], dtype=np.float32)
        
        if self.index + self.iteration == len(self.data):
            done = True
        else:
            done = False

        # Define additional information (optional)
        info = {}

        return np.array(self.state, dtype=np.float32), reward, done, info
    
    def render(self):
        pass
            
    def close(self):
        pass