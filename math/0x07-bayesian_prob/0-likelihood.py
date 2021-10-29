#!/usr/bin/env python3
"Likelihood"
import numpy as np


def likelihood(x, n, P):
    """calculate the likelihood of our data"""

    """
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
    """

    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.min(P) < 0 or np.max(P) > 1:
        raise ValueError('All values in P must be in the range [0, 1]')
    #  binomial distribution
    y = n - x
    f = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(y))
    return f * (P ** x) * ((1 - P) ** y)


if __name__ == "__main__":
    P = np.linspace(0, 1, 11) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))