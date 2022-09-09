#!/usr/bin/env python3
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):

    m = Y.shape[1]
    for i in reversed(range(1, L + 1)):
        w = weights['W' + str(i)]
        b = weights['b' + str(i)]
        a0 = cache['A' + str(i - 1)]
        a = cache['A' + str(i)]
        if i == L:
            dz = a - Y
            W = w
        else:
            d = cache['D' + str(i)]
            da = 1 - (a * a)
            dz = np.matmul(W.T, dz)
            dz = dz * da * d
            dz = dz / keep_prob
            W = w
        dw = np.matmul(a0, dz.T) / m
        db = np.mean(dz, axis=1, keepdims=True)
        weights['W' + str(i)] = w - alpha * dw.T
        weights['b' + str(i)] = b - alpha * db
