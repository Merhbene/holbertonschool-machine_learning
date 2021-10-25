#!/usr/bin/env python3
"""
Correlation
"""


import numpy as np


def correlation(C):
    """ Calculates the correlation matrix """

    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    diag = np.sqrt(np.diag(C))
    return C / np.outer(diag, diag)


if __name__ == '__main__':
    import numpy as np
    correlation = __import__('1-correlation').correlation

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
