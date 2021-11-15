#!/usr/bin/env python3
"""Module contains MultiNormal Class"""


import numpy as np


class MultiNormal():
    """
       Class which represents multivariate normal
        distribution.
    """

    def __init__(self, data):
        """
        Class Constructor
        Args:
            data: numpy.ndarray of shape (d, n)
                n: Number of data points.
                d: Number of dimensions in each data point.
        Public Attributes:
            mean: numpy.ndarray of shape (d, 1)
            cov: numpy.ndarray of shape (d, d
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        #self.mean, self.cov = self.mean_cov(data.T)
        n = data.shape[1]
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul((data - self.mean), (data - self.mean).T) / (n - 1)


    def pdf(self, x):
        """
           Calculates multivariate pdf at data point.
           Args:
            x: numpy.ndarray - shape (d, 1) Data point.
           Return:
            Value of the pdf.
        """
        D = self.mean.shape[0]

        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        x1, x2 = x.shape

        if len(x.shape) != 2 or x1 != D or x2 != 1:
            raise ValueError(
                "x must have the shape ({}, 1)".format(D)
                )

        Px = (2*np.pi)**(D/2)
        Px = 1 / (Px * (np.linalg.det(self.cov)**(1/2)))
        covI = np.linalg.inv(self.cov)
        x_mu = x - self.mean
        dot = np.dot(np.dot(x_mu.T, covI), x_mu)
        return float(Px*np.exp((-1/2)*dot))

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
