#!/usr/bin/env python3
"""updates the weights and biases of a neural network
   using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): 
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m)*(np.matmul(dz, cache["A" + str(i-1)].T) + (lambtha * weights["W" + str(i)]))
        db = (1 / m)*np.sum(dz, axis=1, keepdims=True)
        dA = cache["A" + str(i-1)]*(1 - cache["A"+str(i-1)])
        dz = np.matmul(weights["W" + str(i)].T, dz) * dA
        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * dw)
        weights["b" + str(i)] = weights["b" + str(i)]-(alpha * db)
