#!/usr/bin/env python3
import numpy as np
import csv
import gymnasium as gym
from gym import Env
from gym.spaces import Box
import numpy as np
from typing import Optional

# Open csv
file = open('trainingData/traj1_trainingData.csv')

# Read csv
csvreader = csv.reader(file)

# Store csv data in numpy ndarray
rows = []
for row in csvreader:
    rows.append(row)
file.close()
data = np.array(rows, dtype=np.float32)

# Normalize data
max = np.ndarray.max(data)
min = np.ndarray.min(data)
data = (data - min) / (max - min) 
data = np.concatenate(data)

class ts_forecasting_env(Env):
    def __init__(self, render_mode: Optional[str] = None):
        # Chosen trajectory
        self.trajectory_idx = 1 

        # Number of historical data points
        self.historical_dp = 10

        # Low states
        low = np.zeros([self.historical_dp], dtype=np.float32)
        
        # High states
        high = np.ones([self.historical_dp], dtype=np.float32)

        # Define the action and state spaces
        self.action_space = Box(0, 1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low, high, shape=(self.historical_dp,), dtype=np.float32)

        # Empty array to store chosen actions for the last episode
        self.actions = np.array([])

        # Render
        self.render_mode = render_mode

    def reset(self):
        # Reset the environment to an initial state
        self.iteration = 0

        # self.index = np.random.choice(range(self.historical_dp,len(data) + 1))
        # self.state = np.array(data[self.index-self.historical_dp:self.index], dtype=np.float32)

        self.state = np.array(data[0:self.historical_dp], dtype=np.float32)
        
        return np.array(self.state, dtype=np.float32)
    
    def step(self, action):
        # Get the current state
        self.current_state = self.state

        # Calculate the reward
        reward = -np.abs(self.current_state[self.historical_dp - 1] - action)

        # Store chosen action
        self.chosen_action = action
        if self.render_mode == 'human':
            self.render()

        # Calculate the next state
        self.iteration += 1

        # Check terminal state
        if self.iteration == len(data) - (self.historical_dp - 1):
            self.state = np.array(data[0:self.historical_dp], dtype=np.float32)
            done = True

        else:
            self.state = np.array(data[0 + self.iteration:self.historical_dp + self.iteration], dtype=np.float32)
            done = False

        # Define additional information (optional)
        info = {}

        return np.array(self.state, dtype=np.float32), reward, done, info
    
    def render(self):
        # Error warning
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == 'human':
            # x axis: step
            self.actions = np.append(self.actions, self.chosen_action)
            # self.x.append(self.current_state[0])
            
    def close(self):
        # Save last episode's chosen actions by the agent
        file = np.column_stack([self.actions])
        file_path = "traj" + str(self.trajectory_idx) + "_results.txt"
        np.savetxt(file_path , file)

# Test the env
env = ts_forecasting_env()
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
print(input_dims)
print(n_actions)
episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0
    steps = 0

    if env.render_mode == 'human':
        env.actions = np.array([])

    while not terminated:
        action = env.action_space.sample()
        n_state, reward, terminated, info = env.step(action)
        score += reward
        steps += 1
        print(n_state)
    print('Episode:{} Score:{}'.format(episode, score))
    print('Number of steps:', steps)

    # save last plot
    if env.render_mode == 'human':
        if episode == episodes:
            env.close()
