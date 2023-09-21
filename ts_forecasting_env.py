#!/usr/bin/env python3
import numpy as np
import csv
import gymnasium as gym
from gym import Env
from gym.spaces import Box
import numpy as np
from typing import Optional
import time 

class ts_forecasting_env(Env):
    def __init__(self, historical_dp=10, trajectory=None, data=None, render_mode: Optional[str] = None):
        # Data
        self.data = data

        # Trajectory
        self.trajectory = trajectory
        # Number of historical data points
        self.historical_dp = historical_dp

        # Low states
        low = np.zeros([self.historical_dp], dtype=np.float64)
        
        # High states
        high = np.ones([self.historical_dp], dtype=np.float64)

        # Define the action and state spaces
        self.action_space = Box(0.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = Box(low, high, shape=(self.historical_dp,), dtype=np.float64)

        # Empty array to store chosen actions for the last episode
        self.actions = np.array([])

        # Render
        self.render_mode = render_mode

    def reset(self):
        # Reset the environment to an initial state
        self.iteration = 0

        # Random initial state
        self.index = np.random.choice(range(self.historical_dp,len(self.data)))
        self.state = np.array(self.data[self.index - self.historical_dp:self.index], dtype=np.float64)

        return np.array(self.state, dtype=np.float64)
    
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
        self.state = np.array(self.data[self.index - self.historical_dp + self.iteration:self.index + self.iteration], dtype=np.float64)
        
        if self.index + self.iteration == len(self.data):
            done = True
        else:
            done = False

        # Define additional information (optional)
        info = {}

        return np.array(self.state, dtype=np.float64), reward, done, info
    
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
        file_path = "traj" + str(self.trajectory) + "_results.txt"
        np.savetxt(file_path , file)


# # Test the env
# # Define variables
# TRAJECTORY = 1
# HISTORICAL_DP = 25
# SPLIT_RATE = 0.80

# # Open csv
# file = open('allData/traj' + str(TRAJECTORY) + '_allData.csv')

# # Read csv
# csvreader = csv.reader(file)

# # Store csv data in numpy ndarray
# rows = []
# for row in csvreader:
#     rows.append(row)
# file.close()
# data_ = np.array(rows, dtype=np.float64)

# # Data split
# split_index = round(len(data_) * SPLIT_RATE)
# train_data, test_data = data_[:split_index], data_[split_index:]

# # Normalize data
# max = np.ndarray.max(data_)
# min = np.ndarray.min(data_)
# train_data_norm = (train_data - min) / (max - min)  
# test_data_norm = (test_data - min) / (max - min)

# # Concatenate data
# TRAIN_DATA = np.concatenate(train_data_norm)
# TEST_DATA = np.concatenate(test_data_norm)

# env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)
# input_dims = env.observation_space.shape[0]
# n_actions = env.action_space.shape[0]
# episodes = 10
# start = time.perf_counter()
# for episode in range(1, episodes+1):
#     state = env.reset()
#     terminated = False
#     score = 0
#     steps = 0

#     if env.render_mode == 'human':
#         env.actions = np.array([])

#     for i in range(1000):
#         action = env.action_space.sample()
#         n_state, reward, terminated, info = env.step(action)
#         score += reward
#         steps += 1
#         if terminated:
#             break

#     print('Episode:{} Score:{}'.format(episode, score))
#     print('Number of steps:', steps)

#     # save last plot
#     if env.render_mode == 'human':
#         if episode == episodes:
#             env.close()

# end = time.perf_counter()

# # Time
# print('Elapsed time: ', end - start)