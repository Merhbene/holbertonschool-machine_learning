#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(k) is not int or k < 1:
        return None
    _, d = X.shape
    low = np.amin(X, 0)
    high = np.amax(X, 0)
    cluster_centroids = np.random.uniform(low, high, size=(k, d))
    return cluster_centroids
