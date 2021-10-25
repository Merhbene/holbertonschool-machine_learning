#!/usr/bin/env python3
"""
Mean and Covariance
"""


import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of X"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov


if __name__ == '__main__':
    import numpy as np
    mean_cov = __import__('0-mean_cov').mean_cov

    np.random.seed(0)
    X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15],
                         [-30, 100, -20], [15, -20, 25]], 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
