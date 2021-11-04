#!/usr/bin/env python3
""" Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculate the expectation step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    d = X.shape[1]
    if type(pi) is not np.ndarray or pi.ndim != 1 or not np.isclose(np.sum(pi), 1):
        return None, None
    k = pi.shape[0]
    if type(m) is not np.ndarray or m.ndim != 2 or m.shape != (k, d):
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3 or S.shape != (k, d, d):
        return None, None

    n = X.shape[0]
    k = pi.shape[0]
    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    l = np.sum(g, axis=0, keepdims=True)
    g = g / l
    return g, np.sum(np.log(l))
