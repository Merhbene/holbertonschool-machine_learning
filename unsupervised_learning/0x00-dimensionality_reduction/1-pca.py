#!/usr/bin/env python3
"""PCA"""

import numpy as np


def pca(X, ndim):
    """perform PCA on a dataset"""
    X = X - np.mean(X, axis=0)
    # Singular Value Decomposition:
    _, s, vh = np.linalg.svd(X)
    W = vh.T
    W = W[:, : ndim]
    T = np.dot(X, W)
    return T


if __name__ == "__main__":
    X = np.loadtxt("mnist2500_X.txt")
    print('X:', X.shape)
    print(X)
    T = pca(X, 50)
    print('T:', T.shape)
    print(T)

