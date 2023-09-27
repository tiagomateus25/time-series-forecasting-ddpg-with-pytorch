#!/usr/bin/env python3
from ddpg import Agent
import numpy as np
from ts_forecasting_env import ts_forecasting_env
import time 
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=int, default=1, help="choose trajectory")
args = parser.parse_args()

# Load and prepare data
############################## Define variables #########################################
TRAJECTORY = args.traj    
HISTORICAL_DP = 25 # historical data points (length of state)
SPLIT_RATE = 0.80  # split data into train and test data
#########################################################################################

# Open csv
file = open('allData/traj' + str(TRAJECTORY) + '_allData.csv')

# Read csv
csvreader = csv.reader(file)

# Store csv data in numpy ndarray
rows = []
for row in csvreader:
    rows.append(row)
file.close()
data_ = np.array(rows, dtype=np.float64)
data_ = np.concatenate(data_)

# Considering relevant data only
if TRAJECTORY == 1:
    data_ = data_[5000:14500]
    COLOR = '#FF0000'
elif TRAJECTORY == 2:
    data_ = data_[3900:7100]
    COLOR = '#FF8000'
elif TRAJECTORY == 3:
    data_ = data_[3000:7500]
    COLOR = '#7F00FF'
elif TRAJECTORY == 4:
    data_ = data_[3250:7425]
    COLOR = '#00CC00'
elif TRAJECTORY == 5:
    data_ = data_[3750:7650]
    COLOR = '#00FFFF'
elif TRAJECTORY == 6:
    data_ = data_[2400:17500]
    COLOR = '#FF00FF'

# Data split
split_index = round(len(data_) * SPLIT_RATE)
train_data, test_data = data_[:split_index], data_[split_index:]

# Normalize data
max = np.max(data_)
min = np.min(data_)
TRAIN_DATA = (train_data - min) / (max - min)  
TEST_DATA = (test_data - min) / (max - min)

# Training setup
############################## Define hyper parameters ##################################
LR_ACTOR = 0.001           
LR_CRITIC = 0.003          
TAU = 0.1                    
GAMMA = 0.9                  
BATCH_SIZE = 128
ACTOR_LAYER = 64
CRITIC_LAYER = 62
REPLAY_BUFFER_SIZE = 100000
#########################################################################################

# Call environment
env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)

# Call agent
agent = Agent(alpha=LR_ACTOR, beta=LR_CRITIC, input_dims=[HISTORICAL_DP], tau=TAU, 
            gamma=GAMMA,batch_size=BATCH_SIZE, layer1_size=ACTOR_LAYER, n_actions=1,
            layer2_size=CRITIC_LAYER, max_size=REPLAY_BUFFER_SIZE)

############################## Define training parameters ###############################
EPISODES = 1
MAX_STEPS = 1000
#########################################################################################

# Train the agent 
reward_history = []
average_reward_history = []
episode_list = []
start = time.perf_counter()
for i in range(1, EPISODES + 1):
    obs = env.reset()
    done = False
    reward = 0

    for step in range(MAX_STEPS):
        act = agent.choose_action(obs)
        new_state, step_reward, done, info = env.step(act)
        agent.remember(obs, act, step_reward, new_state, int(done))
        agent.learn()
        reward += step_reward
        obs = new_state
        if done:
            break

    print('episode:', i, 'reward %.6f' % reward, 'steps: ', step)
    reward_history.append(reward)
    average_reward_history.append(np.mean(reward_history[-10:]))
    episode_list.append(i)

end = time.perf_counter()

# Time
print('Elapsed time: ', end - start, ' seconds.')

# Plot training results
plt.figure(1)
plt.plot(episode_list, average_reward_history, color='blue', label='Average Reward')
plt.plot(episode_list,reward_history, color='orange', label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')

# Test the agent
pred = []
for i in range(len(TEST_DATA)):
    state = np.array(TEST_DATA[0 + i:HISTORICAL_DP + i], dtype=np.float64)
    action = agent.choose_action(state)
    pred.append(action)
    if HISTORICAL_DP + i == len(TEST_DATA) - 1:
        break

pred = np.concatenate(pred)
pred = pd.Series(pred)
pred = pred * (max - min) + min
real = pd.Series(test_data[HISTORICAL_DP:])

# Mean absolute error
print('MAE: ', mean_absolute_error(real, pred))

# Mean squared error
print('MSE: ', mean_squared_error(real, pred, squared=False))

# Root mean squared error
print('RMSE: ', mean_squared_error(real, pred, squared=True))

# Coefficient of determination
print('R2: ', r2_score(real, pred))

# Plot error lines
plt.figure(2)
plt.scatter(pred,real,marker = '.', color=COLOR)
plt.plot([-1, 0, 1], [-1, 0, 1], 'k', linewidth=1)
plt.plot([-1, 0, 1], [-1.2, 0, 1.2], 'k--', linewidth=1)
plt.plot([-1, 0, 1], [-0.8, 0, 0.8], 'k--', linewidth=1)
plt.xlabel('Predicted data')
plt.ylabel('Real data')
plt.title('Trajectory ' + str(TRAJECTORY))

# Plot results
plt.figure(3)
plt.plot(real, color='#0000FF', label='Real data')
plt.plot(pred, color=COLOR, label='Predicted data')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.title('Trajectory ' + str(TRAJECTORY))
plt.show()
