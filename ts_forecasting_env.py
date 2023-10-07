#!/usr/bin/env python3
import numpy as np
from gym import Env
from gym.spaces import Box
from typing import Optional
import csv

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
        self.index = np.random.choice(range(self.historical_dp, len(self.data)))
        self.state = np.array(self.data[self.index - self.historical_dp:self.index], dtype=np.float64)

        return np.array(self.state, dtype=np.float64)
    
    def step(self, action):
        # Get the current state
        self.current_state = self.state

        # Calculate the reward
        reward = - abs(self.data[self.index + self.iteration] - action)

        # Store chosen action
        self.chosen_action = action

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
        pass
            
    def close(self):
        pass

# # Open csv
# TRAJECTORY = 1
# HISTORICAL_DP = 25 # historical data points (length of state)
# SPLIT_RATE = 0.80  # split data into train and test data
# file = open('allData/traj' + str(TRAJECTORY) + '_allData.csv')

# # Read csv
# csvreader = csv.reader(file)

# # Store csv data in numpy ndarray
# rows = []
# for row in csvreader:
#     rows.append(row)
# file.close()
# data_ = np.array(rows, dtype=np.float64)
# data_ = np.concatenate(data_)

# # Considering relevant data only
# if TRAJECTORY == 1:
#     data_ = data_[5000:14500]
#     COLOR = '#FF0000'

# # Data split
# split_index = round(len(data_) * SPLIT_RATE)
# train_data, test_data = data_[:split_index], data_[split_index:]

# # Normalize data
# max = np.max(data_)
# min = np.min(data_)
# TRAIN_DATA = (train_data - min) / (max - min)  
# TEST_DATA = (test_data - min) / (max - min)

# # Call environment
# env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)

# ############################## Define training parameters ###############################
# EPISODES = 1000
# MAX_STEPS = 1000
# #########################################################################################

# # Train the agent 
# reward_history = []
# average_reward_history = []
# episode_list = []

# for i in range(1, EPISODES + 1):
#     obs = env.reset()
#     done = False
#     reward = 0

#     for step in range(MAX_STEPS):
#         act = env.action_space.shape[0]
#         new_state, step_reward, done, info = env.step(act)
#         reward += step_reward
#         obs = new_state
#         if done:
#             break
#     print("episode: ", i, "reward: ", reward, "steps: ", step)