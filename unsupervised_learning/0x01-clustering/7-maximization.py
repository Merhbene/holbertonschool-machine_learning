#!/usr/bin/env python3
""" Maximization """
import numpy as np

def maximization(X, g):
    """calculate maximization of all points"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    if type(g) is not np.ndarray or g.ndim != 2 or g.shape[1] != n or not np.allclose(np.sum(g, axis=0), np.ones(g.shape[1])):
        return None, None, None
    k = g.shape[0]
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    nk = np.sum(g, axis=1)
    pi = nk / n
    
    for i in range(k):
        gi = g[i].reshape((-1, 1))
        m[i] = np.sum(gi * X, axis=0) / nk[i]
        Xm = X - m[i]
        S[i] = np.matmul(Xm.T, Xm * gi) / nk[i]
    return pi, m, S
