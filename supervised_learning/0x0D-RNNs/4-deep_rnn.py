#!/usr/bin/env python3
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN"""
    H = [h_0]
    Y = []
    M = X
    for i in range(len(rnn_cells)):
          cell = rnn_cells[i]
          h, y = rnn(cell, M, h_0[i])
          H.append(h)
          Y.append(y)
          M = h
    return np.array(H), np.array(Y)
