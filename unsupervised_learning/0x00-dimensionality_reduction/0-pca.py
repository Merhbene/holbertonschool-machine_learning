#!/usr/bin/env python3
""" PCA"""
import numpy as np


def pca(X, var=0.95):
    """perform PCA on a dataset"""
    # Singular Value Decomposition:
    _, s, vh = np.linalg.svd(X)
    # Return the cumulative sum of the elements
    total_var = np.cumsum(s) / np.sum(s)
    nd = np.argmax(total_var >= var) + 1
    # var is the fraction of the variance that the PCA transformation should maintain
    W = vh.T
    return W[:,:nd]

if __name__ == "__main__":

    np.random.seed(0)
    a = np.random.normal(size=50)
    b = np.random.normal(size=50)
    c = np.random.normal(size=50)
    d = 2 * a
    e = -5 * b
    f = 10 * c

    X = np.array([a, b, c, d, e, f]).T
    m = X.shape[0]
    X_m = X - np.mean(X, axis=0)
    W = pca(X_m)
    T = np.matmul(X_m, W)
    print(T)
    X_t = np.matmul(T, W.T)
    print(np.sum(np.square(X_m - X_t)) / m)