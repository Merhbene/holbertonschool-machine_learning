#!/usr/bin/env python3
"Play"
import gym
import numpy as np


def play(env, Q, max_steps=100):
    "Trained agent play an episode"
    state = env.reset()
    env.render()
    for step in range(max_steps):
        # Choose action with highest Q-value for current state
        action = np.argmax(Q[state, :])
        # Take new action
        state, reward, done, info = env.step(action)
        # Show current state of environment on screen
        env.render()
        if done:
            break
    return reward
