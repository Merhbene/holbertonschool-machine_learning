#!/usr/bin/env python3
"""
This module contains the function np_slice
"""


def np_slice(matrix, axes={}):
    """This returns a slice of an ndarray"""
    for a in sorted(axes.keys()):

        val = axes[a]
        if len(val) == 3:
            start = val[0]
            stop = val[1]
            step = val[2]
        elif len(val) == 2:
            start = val[0]
            stop = val[1]
            step = None
        elif len(val) == 1:
            start = None
            stop = val[0]
            step = None
        else:
            start = None
            stop = None
            step = None

        s = slice(start, stop, step)
        slc = [slice(None)] * len(matrix.shape)
        slc[a] = s
        matrix = matrix[tuple(slc)]

    return matrix
