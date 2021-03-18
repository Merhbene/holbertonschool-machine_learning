#!/usr/bin/env python3
""" Deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN"""
    H = [h_0]
    Y = []
    for xt in X:
        ht = []
        M = xt
        for i in range(len(rnn_cells)):
            h, y = rnn_cells[i].forward(H[-1][i], M)
            ht.append(h)
            M = h
            if i == (len(rnn_cells)-1):
                Y.append(y)
        H.append(ht)
    return np.array(H), np.array(Y)
