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

    def backward(self, h_next, x_t):
        _, i = x_t.shape
        _, h = h_next.shape
        s = np.matmul(h_next, self.Whb[:h]) + np.matmul(x_t, self.Whb[-i:]) + self.bhb
        s = np.tanh(s)
        return s

    def output(self, H):
        """H is a numpy.ndarray of shape (t, m, 2 * h) that contains the concatenated
         hidden states from both directions, excluding their initialized states """
        y = np.matmul(H, self.Wy) + self.by
        y = np.exp(y)
        y = y / np.sum(y, axis=2, keepdims=True)
        return y 
