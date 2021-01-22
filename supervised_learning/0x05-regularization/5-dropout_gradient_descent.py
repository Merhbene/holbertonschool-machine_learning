#!/usr/bin/env python3
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):

    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1 / m)*(np.matmul(dz, cache["A" + str(i-1)].T)) 
        db = (1 / m)*np.sum(dz, axis=1, keepdims=True)
        dA = (1 - cache["A"+str(i-1)] ** 2)
        dz = np.matmul(weights["W" + str(i)].T, dz) * dA
        if i>1:
            dz = np.matmul(weights["W" + str(i)].T, dz) * dA * cache["D"+str(i-1)] / keep_prob
        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * dw)
        weights["b" + str(i)] = weights["b" + str(i)]-(alpha * db)

