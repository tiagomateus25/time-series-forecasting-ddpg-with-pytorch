#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
import time
import csv 
from ddpg import Agent
from ts_forecasting_env import ts_forecasting_env

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=int, default=1, help="choose trajectory")
args = parser.parse_args()

# Load and prepare data
############################## Define variables #########################################
TRAJECTORY = args.traj    
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

# Data split
split_index = round(len(data_) * SPLIT_RATE)
train_data, test_data = data_[:split_index], data_[split_index:]

# Normalize data
max = np.max(data_)
min = np.min(data_)
TRAIN_DATA = (train_data - min) / (max - min)  
TEST_DATA = (test_data - min) / (max - min)

# Agent number
n = 20

# Iteration number
MAXite = 10

# Number of parameters to optimize
nv = 5

# Parameter windows
xmin = np.array([10, 32, 32, 0.001, 0.003])
xmax = np.array([25, 64, 64, 0.003, 0.005])

# Parameters initialization
VARS = np.zeros([nv,1], dtype=np.float64)
X = np.zeros([nv,n], dtype=np.float64)

for i in range(n):
    X[:3,i] = np.round(xmin[:3] + (xmax[:3] - xmin[:3]) * np.random.rand(3))
    X[3:,i] = np.round(xmin[3:] + (xmax[3:] - xmin[3:]) * np.random.rand(2), decimals=3)

J = np.zeros([1,n])

start = time.perf_counter()
for k in range(n):
    start1 = time.perf_counter()
    VARS = X[:,k]

    # Training setup
    ############################## Define hyper parameters ##################################
    LR_ACTOR = VARS[3]        
    LR_CRITIC = VARS[4]          
    TAU = 0.1                    
    GAMMA = 0.9                  
    BATCH_SIZE = int(VARS[2])
    ACTOR_LAYER = int(VARS[1])
    CRITIC_LAYER = ACTOR_LAYER
    REPLAY_BUFFER_SIZE = 100000
    HISTORICAL_DP = int(VARS[0]) # historical data points (length of state)
    print(HISTORICAL_DP, ACTOR_LAYER, BATCH_SIZE, LR_ACTOR, LR_CRITIC)
    #########################################################################################

    # Call environment
    env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)

    # Call agent
    agent = Agent(alpha=LR_ACTOR, beta=LR_CRITIC, input_dims=[HISTORICAL_DP], tau=TAU, 
                gamma=GAMMA,batch_size=BATCH_SIZE, layer1_size=ACTOR_LAYER, n_actions=1,
                layer2_size=CRITIC_LAYER, max_size=REPLAY_BUFFER_SIZE)

    ############################## Define training parameters ###############################
    EPISODES = 15
    MAX_STEPS = 1000
    #########################################################################################

    np.random.seed(0)

    # Train the agent 
    for i in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        reward = 0

        for step in range(MAX_STEPS):
            act = agent.choose_action(obs)
            new_state, step_reward, done, _ = env.step(act)
            agent.remember(obs, act, step_reward, new_state, int(done))
            agent.learn()
            reward += step_reward
            obs = new_state
            if done:
                break

    # Test the agent
    pred = []
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

    J[0,k] = mean_absolute_error(real, pred)

    end1 = time.perf_counter()
    print('Agent: ', k, ' Elapsed time: ', end1 - start1, ' seconds.')
    
end = time.perf_counter()
print('Total elapsed time: ', end - start, ' seconds.')

Jmin_value = np.min(J)
idx = np.argmin(J)
XBEST = X[:,idx]
JBEST = Jmin_value

for ite in range(MAXite):
    Jmin_value = np.min(J)
    idx = np.argmin(J)
    if Jmin_value < JBEST:
        XBEST = X[:,idx]
        JBEST = Jmin_value
        VARS = XBEST

    for i in range(n):
        X[:3,i] = X[:3,i] + np.round(np.random.rand(3) * (XBEST[:3].T - X[:3, i]))
        X[3:,i] = X[3:,i] + np.round(np.random.rand(2) * (XBEST[3:].T - X[3:, i]), decimals=3)

    print('Iteration: ', ite)
    print('Best parameters: ', XBEST)
    print('MAE: ', JBEST)

    R = (1/JBEST)/np.sum(1/J)

    for i in range(n):
        if i == idx:
            continue

        if np.linalg.norm(XBEST.T - X[:, i]) < R:
            X[:3,i] = np.round(xmin[:3] + (xmax[:3] - xmin[:3]) * np.random.rand(3))
            X[3:,i] = np.round(xmin[3:] + (xmax[3:] - xmin[3:]) * np.random.rand(2), decimals=3)

        if np.array_equal(X[:,i], XBEST):
            X[:3,i] = np.round(xmin[:3] + (xmax[:3] - xmin[:3]) * np.random.rand(3))
            X[3:,i] = np.round(xmin[3:] + (xmax[3:] - xmin[3:]) * np.random.rand(2), decimals=3)

    for k in range(n):
        VARS = X[:,k]

        # Training setup
        ############################## Define hyper parameters ##################################
        LR_ACTOR = VARS[3]        
        LR_CRITIC = VARS[4]          
        TAU = 0.1                    
        GAMMA = 0.9                  
        BATCH_SIZE = int(VARS[2])
        ACTOR_LAYER = int(VARS[1])
        CRITIC_LAYER = ACTOR_LAYER
        REPLAY_BUFFER_SIZE = 100000
        HISTORICAL_DP = int(VARS[0]) # historical data points (length of state)
        #########################################################################################

        # Call environment
        env = ts_forecasting_env(historical_dp=HISTORICAL_DP, data=TRAIN_DATA)

        # Call agent
        agent = Agent(alpha=LR_ACTOR, beta=LR_CRITIC, input_dims=[HISTORICAL_DP], tau=TAU, 
                    gamma=GAMMA,batch_size=BATCH_SIZE, layer1_size=ACTOR_LAYER, n_actions=1,
                    layer2_size=CRITIC_LAYER, max_size=REPLAY_BUFFER_SIZE)

        ############################## Define training parameters ###############################
        EPISODES = 15
        MAX_STEPS = 1000
        #########################################################################################

        np.random.seed(0)

        # Train the agent 
        for i in range(1, EPISODES + 1):
            obs = env.reset()
            done = False
            reward = 0

            for step in range(MAX_STEPS):
                act = agent.choose_action(obs)
                new_state, step_reward, done, _ = env.step(act)
                agent.remember(obs, act, step_reward, new_state, int(done))
                agent.learn()
                reward += step_reward
                obs = new_state
                if done:
                    break

        # Test the agent
        pred = []
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

        J[0,k] = mean_absolute_error(real, pred)
        print('Agent: ', k)