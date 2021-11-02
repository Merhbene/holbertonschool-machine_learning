#!/usr/bin/env python3
"""K-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
  "performs K-means on a dataset"
  _, d = X.shape
  mins = np.min(X, axis=0)
  maxs = np.max(X, axis=0)
  
  centroids = np.random.uniform(mins, maxs, size=(k, d))
  new_centroids = centroids.copy()
  for i in range(iterations):
      
      dist = np.square(X[:, None, :] - centroids[None, :, :]).sum(axis=-1)
      clss = np.argmin(dist, axis=1)
      

      for c in range(k):
          indices = np.argwhere(clss == c)
          #centroids[indices] = np.mean(x[indices], axis=0)
          if len(X[indices]) > 0:
              new_centroids[c] = np.mean(X[indices], axis=0)
          else:
              new_centroids[c] = np.random.uniform(mins, maxs, size=d)
          

      if np.array_equal(centroids, new_centroids):
          break
      centroids = new_centroids.copy()
  return centroids, clss
