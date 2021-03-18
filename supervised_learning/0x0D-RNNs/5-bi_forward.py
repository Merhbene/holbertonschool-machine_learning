#!/usr/bin/env python3
"""bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell:
    def __init__(self, i, h, o):
        """public instance attributes"""
        self.Whf = np.random.randn(h + i, h)
        self.Whb =  np.random.randn(h + i, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf =  np.zeros((1, h))
        self.bhb =  np.zeros((1, h))
        self.by =  np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in the forward direction for one time step """
        _, i = x_t.shape
        _, h = h_prev.shape
        s = np.matmul(h_prev, self.Whf[:h]) + np.matmul(x_t, self.Whf[-i:]) + self.bhf
        s = np.tanh(s)
        return s
