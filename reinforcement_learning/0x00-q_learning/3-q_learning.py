#!/usr/bin/env python3
"Q-learning"
import gym
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
      "Performs Q-learning:"
      total_rewards = []
      max_epsilon = epsilon

      # Q learning algorithm
      for episode in range(episodes):

          # initialize new episode params
          state = env.reset()
          done = False 
          rewards_current_episode = 0

          for step in range(max_steps):

              # exploration-exploitation trade-off
              action = epsilon_greedy(Q, state, epsilon)

              new_state, reward, done, info = env.step(action)

              # Checking if I fell in a hole
              if done and reward == 0 :
                  reward = -1 

              # update  Q-table for Q(s, a)
              Q[state, action] = Q[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[new_state, :]))

              state = new_state
              rewards_current_episode += reward

              if done: #step in a hole or reach the goal
                  break

          # exploration rate decay 
          epsilon = min_epsilon + (max_epsilon -  min_epsilon) * np.exp(- epsilon_decay * episode)

          total_rewards.append(rewards_current_episode)

      return Q, total_rewards
