#!/usr/bin/env python3
"Initialize Q-table"
import gym


def q_init(env):
    " Initializes the Q-table"
    action_space_size = env.action_space.n # nb of possible actions
    state_space_size = env.observation_space.n # nb of states
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
