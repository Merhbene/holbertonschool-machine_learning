#!/usr/bin/env python3
"""K-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    "performs K-means on a dataset"
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int  or k < 1:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    _, d = X.shape
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    
    centroids = np.random.uniform(mins, maxs, size=(k, d))
    new_centroids = centroids.copy()
    for i in range(iterations):
        
        dist = np.square(X[:, None, :] - centroids[None, :, :]).sum(axis=-1)
        clss = np.argmin(dist, axis=1)
        

        for c in range(k):
            indices = np.argwhere(clss == c).reshape(-1)
            #centroids[indices] = np.mean(x[indices], axis=0)
            if len(X[indices]) > 0:
                new_centroids[c] = np.mean(X[indices], axis=0)
            else:
                new_centroids[c] = np.random.uniform(mins, maxs, size=d)
            

        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids.copy()
    return centroids, clss

