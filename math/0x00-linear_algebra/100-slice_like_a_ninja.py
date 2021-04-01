#!/usr/bin/env python3
"""
This module contains the function np_slice
"""


def np_slice(matrix, axes={}):
    """This returns a slice of an ndarray"""
    last = -1
    mat_slice = []
    for axis in sorted(axes):
        mat_slice += [slice(None)]* (axis - last - 1) + [slice(*axes[axis])]
        last = axis
    return matrix[mat_slice]
