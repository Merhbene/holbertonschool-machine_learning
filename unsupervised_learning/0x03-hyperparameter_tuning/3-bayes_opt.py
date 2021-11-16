#!/usr/bin/env python3`
"""Hyperparameter tuning module"""
import numpy as np
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
        self.X_s = np.expand_dims(X_s, axis=1) # or X_s.reshape(-1, 1)
