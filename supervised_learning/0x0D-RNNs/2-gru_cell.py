#!/usr/bin/env python3
import numpy as np
""" GRU Cell """


class GRUCell:
    def __init__(self, i, h, o):
         self.Wz = np.random.randn(h+i, h) # for the update gate
         self.Wr = np.random.randn(h+i, h) # for the reset gate
         self.Wh = np.random.randn(h+i, h) # for the intermediate hidden state
         self.Wy = np.random.randn(h, o) # for the output
         self.bz = np.zeros((1,h))
         self.br = np.zeros((1,h))
         self.bh = np.zeros((1,h))
         self.by = np.zeros((1,o))

    def forward(self, h_prev, x_t):
        "Perform forward propagation for one time step"
        m, i = x_t.shape
        m, h = h_prev.shape

        z = np.matmul(h_prev, self.Wz[:h]) + np.matmul(x_t, self.Wz[-i:]) + self.bz
        z = (1 / (1 + np.exp(-z)))

        r = np.matmul(h_prev, self.Wr[:h]) + np.matmul(x_t, self.Wr[-i:]) + self.br
        r = (1 / (1 + np.exp(-r)))

        h = np.matmul(x_t, self.Wh[-i:]) + np.matmul(h_prev * r, self.Wh[:h]) + self.bh
        h = np.tanh(h)

        h_next = ((1 - z) * h_prev )+ (z * h)

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y)
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, y
