#!/usr/bin/env python3
"""Q affinities"""
import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities"""
    # Y contain the low dimensional transformation of X
    D = np.square(Y[:, None, :] - Y[None, :, :]).sum(axis=-1)
    num = (1 + D) ** (-1)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
