#!/usr/bin/env python3
"calculate the cost of a neural network with L2 regularization"
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    "cost is the cost of the network without L2 regularization"
    reg = 0
    for i in range(1, L+1):
        w = weights['W' + str(i)]
        reg += np.linalg.norm(w)
    reg = (lambtha * reg) / (2 * m)
    return cost + reg
