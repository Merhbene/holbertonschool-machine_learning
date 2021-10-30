#!/usr/bin/env python3
"""Gradients"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """calculates the gradients of Y"""
    n, ndims = Y.shape
    # dY = np.zeros((n, ndims))
    Q, num = Q_affinities(Y)
    PQ = P - Q
    PQ = PQ[:, :, None]
    num = num[:, :, None]
    Y = Y[:, None, :] - Y[None, :, :] 
    # or np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    dY = PQ * Y * num
    return dY.sum(axis=1), Q

