#!/usr/bin/env python3`
"""Hyperparameter tuning module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization class
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
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
