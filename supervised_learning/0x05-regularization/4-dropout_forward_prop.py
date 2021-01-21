#!/usr/bin/env python3
"A Dropout function"
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    "conduct forward propagation using Dropout"
    cache = {'A0': X}

    for i in range(L):
        W_key = "W{}".format(i + 1)
        b_key = "b{}".format(i + 1)
        A_key_prev = "A{}".format(i)
        A_key_forw = "A{}".format(i + 1)
        D_key = "D{}".format(i + 1)

        Z = np.matmul(weights[W_key], cache[A_key_prev]) + weights[b_key]
        drop = np.random.binomial(1, keep_prob, size=Z.shape)

        if i == L - 1:
            t = np.exp(Z)
            cache[A_key_forw] = (t / np.sum(t, axis=0, keepdims=True))
        else:
            cache[A_key_forw] = np.tanh(Z)
            cache[D_key] = drop
            cache[A_key_forw] = (cache[A_key_forw] * cache[D_key]) / keep_prob
    return cache
