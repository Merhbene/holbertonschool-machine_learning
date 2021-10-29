#!/usr/bin/env python3
" Intersection "
import numpy as np


def intersection(x, n, P, Pr):
    """calculate the intersection given our data and priors"""

    """
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    """
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
    return L * Pr

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
