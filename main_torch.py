#!/usr/bin/env python3
from ddpg_torch import Agent
import numpy as np
from ts_forecasting_env import ts_forecasting_env
import time 
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Load and prepare data
############################## Define variables ##############################
TRAJECTORY = 1
HISTORICAL_DP = 25
SPLIT_RATE = 0.80
##############################################################################

# Open csv
file = open('allData/traj' + str(TRAJECTORY) + '_allData.csv')

# Read csv
csvreader = csv.reader(file)

# Store csv data in numpy ndarray
rows = []
for row in csvreader:
    rows.append(row)
file.close()
data_ = np.array(rows, dtype=np.float32)

# Data split
split_index = round(len(data_) * SPLIT_RATE)
train_data, test_data = data_[:split_index], data_[split_index:]

# Normalize data
max = np.ndarray.max(data_)
min = np.ndarray.min(data_)
train_data_norm = (train_data - min) / (max - min)  
test_data_norm = (test_data - min) / (max - min)

# Concatenate data
TRAIN_DATA = np.concatenate(train_data_norm)
TEST_DATA = np.concatenate(test_data_norm)

# Training setup
############################## Define hyper parameters ##############################
LR_ACTOR = 0.0001
LR_CRITIC = 0.0003
TAU = 0.1
GAMMA = 0.9
BATCH_SIZE = 256
ACTOR_LAYER = 64
CRITIC_LAYER = 64
#####################################################################################

env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)
agent = Agent(alpha=LR_ACTOR, beta=LR_CRITIC, input_dims=[HISTORICAL_DP], tau=TAU, env=env, gamma=GAMMA,
              batch_size=BATCH_SIZE, layer1_size=ACTOR_LAYER, layer2_size=CRITIC_LAYER, n_actions=1, max_size=100000)

np.random.seed(0)

episodes = 60
max_steps = 1000
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

    print('episode:', i, 'reward %.2f' % reward, 'steps: ', step)
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
pred = []
for i in range(len(TEST_DATA[:750])):
    state = np.array(TEST_DATA[0 + i:HISTORICAL_DP + i], dtype=np.float32)
    action = agent.choose_action(state)
    pred.append(action)
    if HISTORICAL_DP + i == len(TEST_DATA) - 1:
        break

pred = pd.Series(pred)
pred = pred*(max-min)+min
test_data = np.concatenate(test_data[:750])
actual = pd.Series(test_data[HISTORICAL_DP:])

plt.figure(2)
plt.scatter(pred,actual,marker = '.')
plt.plot(np.linspace(-0.001,0.002), np.linspace(-0.001,0.002), color='red')
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')

plt.figure(3)
plt.plot(actual, color='blue', label='real data')
plt.plot(pred, color='orange', label='predicted data')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.show()