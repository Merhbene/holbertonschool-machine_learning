#!/usr/bin/env python3
"""
Contains the class MultiNormal
"""


import numpy as np


class MultiNormal:
    """represents a multinormal distribution"""
    def __init__(self, data):
        """initializes a multinormal instance"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul((data - self.mean), (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """calculate the pdf of the multinormal distribution"""
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if x.shape != self.mean.shape:
            raise ValueError('x must have the shape {}'.format(self.mean.shape))
        d = x.shape[0]
        xm = x - self.mean
        exponent = - 0.5 * np.matmul(xm.T, np.matmul(np.linalg.inv(self.cov), xm))
        return np.exp(exponent[0, 0]) / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))


if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
