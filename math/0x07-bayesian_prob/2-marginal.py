#!/usr/bin/env python3
" Marginal probability "
import numpy as np


def marginal(x, n, P, Pr):
    """calculate the marginal probilility of our data"""
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.min(P) < 0 or np.max(P) > 1:
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.min(Pr) < 0 or np.max(Pr) > 1:
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    y = n - x
    f = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(y))
    L = f * (P ** x) * ((1 - P) ** y)
    I = L * Pr
    return np.sum(I)

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
