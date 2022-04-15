#!/usr/bin/env python3
"""Gaussian Process Class"""
import numpy as np


class GaussianProcess:
    "represents a noiseless 1D Gaussian process"
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        "class constructor"
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        # k crepresent the current covariance kernel matrix for the Gaussian process
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        "calculates the covariance kernel matrix between two matrices"
        sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp((-1 / (2 * self.l ** 2)) * sqdist)


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GaussianProcess(X_init, Y_init, l=0.6, sigma_f=2)
    print(gp.X is X_init)
    print(gp.Y is Y_init)
    print(gp.l)
    print(gp.sigma_f)
    print(gp.K.shape, gp.K)
    print(np.allclose(gp.kernel(X_init, X_init), gp.K))