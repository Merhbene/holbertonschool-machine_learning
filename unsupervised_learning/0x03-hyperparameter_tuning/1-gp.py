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

    def predict(self, X_s):
        "predicts the mean and standard deviation of points in a Gaussian process"

        k_s = self.kernel(self.X, X_s)
        k_ss = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(self.K)

        mu = np.dot(np.dot(k_s.T, k_inv), self.Y)
        sigma = k_ss - np.dot(np.dot(k_s.T, k_inv), k_s)
        return mu.reshape(-1), sigma.diagonal()


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GaussianProcess(X_init, Y_init, l=0.6, sigma_f=2)
    X_s = np.random.uniform(-np.pi, 2*np.pi, (10, 1))
    mu, sig = gp.predict(X_s)
    print(mu.shape, mu)
    print(sig.shape, sig)
