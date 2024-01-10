#!/usr/bin/env python3
from ddpg import Agent
import numpy as np
from ts_forecasting_env import ts_forecasting_env
import time 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch


# Load and prepare data
############################## Define variables #########################################  
HISTORICAL_DP = 24 # historical data points (length of state)
SPLIT_RATE = 0.80  # split data into train and test data
DATA = pd.read_csv('INSERT DATA PATH', header=None)
data = DATA.iloc[:,0] # all data from the first column
data = data.values.astype('float32')
#########################################################################################

# Normalize data
max = np.max(data)
min = np.min(data)

# Data split
split_index = round(len(data) * SPLIT_RATE)
train_data, test_data = data[:split_index], data[split_index:]

# Normalize data
TRAIN_DATA = (train_data - min) / (max - min)  
TEST_DATA = (test_data - min) / (max - min)

# Training setup
############################## Define hyper parameters ##################################
LR_ACTOR = 0.003           
LR_CRITIC = 0.005          
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
EPISODES = 30
MAX_STEPS = 1000
#########################################################################################

np.random.seed(0)

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

# Elapsed time
print('Elapsed time: ', end - start, ' seconds.')

# Plot training results
plt.figure(1)
plt.plot(episode_list, average_reward_history, color='blue', label='Average Reward')
plt.plot(episode_list,reward_history, color='orange', label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')

# Test the agent
pred = []
with torch.no_grad():
    for i in range(len(TEST_DATA)):
        state = np.array(TEST_DATA[0 + i:HISTORICAL_DP + i], dtype=np.float64)
        action = agent.choose_action(state)
        pred.append(action)
        if HISTORICAL_DP + i == len(TEST_DATA):
            break

pred = np.concatenate(pred)
pred = pd.Series(pred)
pred = pred * (max - min) + min
real = pd.Series(test_data[HISTORICAL_DP - 1:])

# Evaluation metrics

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
plt.scatter(pred,real,marker = '.')
plt.plot([-1, 0, 1], [-1, 0, 1], 'k', linewidth=1)
plt.plot([-1, 0, 1], [-1.2, 0, 1.2], 'k--', linewidth=1)
plt.plot([-1, 0, 1], [-0.8, 0, 0.8], 'k--', linewidth=1)
plt.xlabel('Predicted data')
plt.ylabel('Real data')
plt.title('Error lines plot')

# Plot prediction results
plt.figure(3)
plt.plot(real, label='Real data')
plt.plot(pred, label='Predicted data')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.title('Prediction results')
plt.show()
