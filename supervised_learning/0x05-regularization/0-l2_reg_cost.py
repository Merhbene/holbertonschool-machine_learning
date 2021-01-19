#!/usr/bin/env python3
import numpy as np


"calculate the cost of a neural network with L2 regularization"
def l2_reg_cost(cost, lambtha, weights, L, m):
    reg = 0
    "cost is the cost of the network without L2 regularization"
    for i in range(1,L+1):
        w = weights['W' + str(i)]
        reg += np.linalg.norm(w)
    reg = (lambtha  * reg) / (2 * m)
    "return the cost of the network accounting for L2 regularization"
    return cost + reg
