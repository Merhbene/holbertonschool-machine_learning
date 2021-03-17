#!/usr/bin/env python3
import numpy as np
""" RNN Cell """


class RNNCell:
    """represent a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Wh and bh are for the concatenated hidden state and input data
        Wy and by are for the output
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
        h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        m is the batche size for the data
        """
        m, i = x_t.shape
        m, h = h_prev.shape
        s = np.matmul(h_prev, self.Wh[:h]) + np.matmul(x_t, self.Wh[-i:]) + self.bh
        # or s = np.matmul(np.concatenate([h_prev, x_t], axis=1), self.Wh)+ self.bh
        s = np.tanh(s)

        y = np.matmul(s, self.Wy) + self.by
        y = np.exp(y)
        y = y / np.sum(y, axis=1, keepdims=True)

        return s, y
