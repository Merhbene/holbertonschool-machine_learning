#!/usr/bin/env python3`
"""Hyperparameter tuning module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization class
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        "Class constructor"
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize

        min, max = bounds
        X_s = np.linspace(min, max, ac_samples)
        self.X_s = np.expand_dims(X_s, axis=1)  # or X_s.reshape(-1, 1)

    def acquisition(self):
        "calculates the next best sample location"
        "Expected Improvement acquisition function"
        "Returns: X_next, EI"

        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        Z = np.where(sigma == 0, 0, imp / sigma)
        # Z = imp / sigma
        ei = np.where(sigma == 0, 0, imp * norm.cdf(Z) + sigma * norm.pdf(Z))
        # ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei


    def optimize(self, iterations=100):
        "optimizes the black-box function"
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            i_opt = np.argmin(self.gp.Y)
        else:
            i_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[i_opt]
        Y_opt = self.gp.Y[i_opt]

        return X_opt, Y_opt


BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print('All sample inputs:', bo.gp.X)
