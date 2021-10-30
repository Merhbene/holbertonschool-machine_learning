#!/usr/bin/env python3
"t_SNE"
import numpy as np


pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Calculates the t-SNE transformation"""
    X = pca(X, idims).real
    n = X.shape[0]
    momentum = 0.5
    final_momentum = 0.8
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    P = P_affinities(X, 1e-5, perplexity)
    P = 4 * P
    for iter in range(iterations):
        dY, Q = grads(Y, P)
        if iter == 20:
            momentum = final_momentum
        iY = momentum * iY - lr * dY
        Y = Y + iY
        Y = Y - np.mean(Y, 0)
        if (iter + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(iter + 1, C))
        if iter + 1 == 100:
            P = P / 4
    return Y
