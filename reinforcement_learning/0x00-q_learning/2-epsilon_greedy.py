#!/usr/bin/env python3
"Epsilon Greedy"
import gym
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    # Uses epsilon-greedy to determine the next action
    p = np.random.uniform(0,1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(Q.shape[1])
    return action
