#!/usr/bin/env python3

import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """monte carlo"""
    states = V.shape[0]
    # evaluate episodes
    for i in range(episodes):
        s = env.reset()
        episode = []
        for j in range(max_steps):
            action = policy(s)
            s_new, reward, done, info = env.step(action)
            episode.append([s, action, reward, s_new])
            if done:
                break
            s = s_new
        episode = np.array(episode, dtype=int)
        G = 0
        for j, step in enumerate(episode[::-1]):
            s, action, reward, s_next = step
            G = gamma * G + reward
            if s not in episode[:i, 0]:
                V[s] = V[s] + alpha * (G - V[s])
    return V