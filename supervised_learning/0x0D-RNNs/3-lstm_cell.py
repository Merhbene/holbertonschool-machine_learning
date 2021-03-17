
#!/usr/bin/env python3
import numpy as np
""" GRU Cell """


class LSTMCell:
    def __init__(self, i, h, o):
         self.Wf = np.random.randn(h+i, h) # for the forget gate
         self.Wu = np.random.randn(h+i, h) # for the update gate
         self.Wc = np.random.randn(h+i, h) # for the intermediate cell state
         self.Wo = np.random.randn(h+i, h) # for the output gate
         self.Wy = np.random.randn(h, o) # for the outputs

         self.bf = np.zeros((1,h))
         self.bu = np.zeros((1,h))
         self.bc = np.zeros((1,h))
         self.bo = np.zeros((1,h))
         self.by = np.zeros((1,o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step"""
        _, i = x_t.shape
        _, h = h_prev.shape

        f = np.matmul(h_prev, self.Wf[:h]) + np.matmul(x_t, self.Wf[-i:]) + self.bf
        f = (1 / (1 + np.exp(-f)))

        u = np.matmul(h_prev, self.Wu[:h]) + np.matmul(x_t, self.Wu[-i:]) + self.bu
        u = (1 / (1 + np.exp(-u)))

        c = np.matmul(h_prev, self.Wc[:h]) + np.matmul(x_t, self.Wc[-i:]) + self.bc
        c = np.tanh(c)
        c_next = (c_prev * f) + (c * u)

        o = np.matmul(h_prev, self.Wo[:h]) + np.matmul(x_t, self.Wo[-i:]) + self.bo
        o = (1 / (1 + np.exp(-o)))

        h_next = np.tanh(c_next) * o

        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y)
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, c_next,  y 
