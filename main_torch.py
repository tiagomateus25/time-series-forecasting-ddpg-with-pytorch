#!/usr/bin/env python3
from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
from ts_forecasting_env import ts_forecasting_env

env = ts_forecasting_env(render_mode='human')
agent = Agent(alpha=0.001, beta=0.003, input_dims=[10], tau=0.1, env=env,
              batch_size=128, layer1_size=64, layer2_size=300, n_actions=1)

#agent.load_models()
np.random.seed(0)

episodes = 300
score_history = []
for i in range(episodes):
    obs = env.reset()
    # done = False
    score = 0

    # Render
    if env.render_mode == 'human':
        env.actions = np.array([])

    for step in range(1000):
    # while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    score_history.append(score)

    # save last plot
    if env.render_mode == 'human':
        if i == episodes-1:
            env.close()

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score),
        #   'trailing 100 episo avg %.3f' % np.mean(score_history[-100:]))

filename = 'results.png'
plotLearning(score_history, filename, window=100)