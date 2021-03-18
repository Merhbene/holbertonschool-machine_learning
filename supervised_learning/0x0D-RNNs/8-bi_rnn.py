#!/usr/bin/env python3
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    "perform forward propagation for a bidirectional RNN"
    t, _, _ = X.shape
    H1 = [h_0]
    H2 = [h_t]
    for step in range(t):
          h1 = bi_cell.forward(H1[-1], X[step])
          h2 = bi_cell.backward(H2[-1], X[t - step - 1])
          H1.append(h1)
          H2.append(h2)
    H1 = np.array(H1[1:])
    H2.reverse()
    H2 = np.array(H2[:-1])
    """H is a numpy.ndarray of shape (t, m, 2 * h) that contains the concatenated hidden 
    states from both directions, excluding their initialized states""" 
    H = np.concatenate((H1, H2), axis=2)
    Y = bi_cell.output(H)
    return H, Y
