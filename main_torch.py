#!/usr/bin/env python3
from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
from ts_forecasting_env import ts_forecasting_env
import time 
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Define trajectory and historical data points
trajectory = 1
historical_dp = 7

env = ts_forecasting_env(historical_dp, trajectory)
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[historical_dp], tau=0.1, env=env,
              batch_size=32, layer1_size=32, layer2_size=32, n_actions=1, max_size=100000)

np.random.seed(0)

episodes = 400
max_steps = 100
reward_history = []
average_reward_history = []
episode_list = []

start = time.perf_counter()
for i in range(1, episodes + 1):
    obs = env.reset()
    done = False
    reward = 0

    # Render
    if env.render_mode == 'human':
        env.actions = np.array([])

    for step in range(max_steps):
        act = agent.choose_action(obs)
        new_state, step_reward, done, info = env.step(act)
        agent.remember(obs, act, step_reward, new_state, int(done))
        agent.learn()
        reward += step_reward
        obs = new_state
        if done:
            break

    print('episode:', i, 'reward %.2f' % reward, 'trailing 10 episode avg %.3f' % np.mean(reward_history[-10:]))
    reward_history.append(reward)
    average_reward_history.append(np.mean(reward_history[-10:]))
    episode_list.append(i)

    # save last plot
    if env.render_mode == 'human':
        if i == episodes:
            env.close()

end = time.perf_counter()

# Time
print('Elapsed time: ', end - start, ' seconds.')

# Plot training
plt.figure(1)
plt.plot(episode_list, average_reward_history, color='blue', label='Average Reward')
plt.plot(episode_list,reward_history, color='orange', label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')


# Test agent

# Open csv
file = open('testingData/traj' + str(trajectory) + '_testingData.csv')

# Read csv
csvreader = csv.reader(file)

# Store csv data in numpy ndarray
rows = []
for row in csvreader:
    rows.append(row)
file.close()
data_ = np.array(rows, dtype=np.float32)

# Normalize data
max = np.ndarray.max(data_)
min = np.ndarray.min(data_)
data = (data_ - min) / (max - min) 

# Concatenate data
data = np.concatenate(data)

pred = []
for i in range(len(data)):
    state = np.array(data[0 + i:historical_dp + i], dtype=np.float32)
    action = agent.choose_action(state)
    pred.append(action)
    if historical_dp + i == len(data) - 1:
        break

pred = pd.Series(pred)
pred = pred*(max-min)+min
data_ = np.concatenate(data_)
actual = pd.Series(data_[historical_dp:])

plt.figure(2)
plt.scatter(pred,actual,marker = '.')
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')
plt.show()