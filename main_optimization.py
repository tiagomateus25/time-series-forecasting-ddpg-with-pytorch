#!/usr/bin/env python3
from ddpg import Agent
import numpy as np
from ts_forecasting_env import ts_forecasting_env
import time 
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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

# Run LSTM with tuning configurations
def tune_lstm(config):   
    # Training setup
    ############################## Define hyper parameters ##################################
    LR_ACTOR = config["a_lr"]        
    LR_CRITIC = config["c_lr"]          
    TAU = 0.1                    
    GAMMA = 0.9                  
    BATCH_SIZE = config["bs"]
    ACTOR_LAYER = config["layer"]
    CRITIC_LAYER = config["layer"]
    REPLAY_BUFFER_SIZE = 100000
    HISTORICAL_DP = config["hdp"] # historical data points (length of state)
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
    real = pd.Series(test_data[HISTORICAL_DP:])

    # Report result to tuner
    # MAE
    tune.report(mean_accuracy=mean_absolute_error(real, pred))
    # # MSE
    # tune.report(mean_accuracy=mean_squared_error(real, pred, squared=False))


# Tuner configurations
config = {
    "a_lr": tune.grid_search([0.001, 0.002, 0.003, 0.004, 0.005]),
    "c_lr": tune.grid_search([0.001, 0.002, 0.003, 0.004, 0.005]),
    "bs": tune.grid_search([2 ** i for i in range(5,8)]),
    "layer": tune.grid_search([2 ** i for i in range(5,8)]),
    "hdp": tune.grid_search([10, 15, 25]),
}

# Run tuner
analysis = tune.run(
    tune_lstm,
    resources_per_trial={"cpu": 12, "gpu": 1},
    config=config,
    mode="min"
    )

print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

df = analysis.dataframe()

