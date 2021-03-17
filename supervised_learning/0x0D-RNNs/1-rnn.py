#!/usr/bin/env python3
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN"""
    t, _, _ = X.shape
    H = [h_0]
    Y = []
    for step in range(t):
          h, y = rnn_cell.forward(H[-1], X[step])
          H.append(h)
          Y.append(y)
    return np.array(H), np.array(Y)
