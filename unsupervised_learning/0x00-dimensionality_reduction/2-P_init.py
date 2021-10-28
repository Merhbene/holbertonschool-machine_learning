#!/usr/bin/env python3
"""Initialize t-SNE"""

import numpy as np


def P_init(X, perplexity):
    "initializes all variables required to calculate the P affinities in t-SNE"
    n = len(X)
    # the P affinities
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    #  Shannon entropy : Base-2 logarithm of perplexity
    H = np.log2(perplexity)
    # D calculates the squared pairwise distance between two data points
    D = np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
    """
    # or:
    for i in range(n):
        for j in range(i):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :]) ** 2
    D += D.T

    # or:
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :]) ** 2
    np.fill_diagonal(D,0)

    # or:
    X1 = X[np.newaxis, :, :]
    X2 = X[:, np.newaxis, :]
    D = np.sum(np.square(X1 - X2), axis=2)
    """
    return D, P, betas, H
