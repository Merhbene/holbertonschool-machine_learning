#!/usr/bin/env python3
"""
This module contains two funtions: matrix_shape and cat_matrices
"""


def matrix_shape(matrix):
    """Calculates the shape of matrix"""
    shape = []
    while type(matrix) is list and len(matrix) > 0:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis and returns the result,
    or None on failure"""
    matrix = []

    if type(mat1) is not type(mat2) or type(mat1) is not list:
        return None
    if axis == 0:
        s1 = matrix_shape(mat1)
        s2 = matrix_shape(mat2)
        if len(s1) != len(s2) or (len(s1) > 0 and s1[1:] != s2[1:]):
            return None
        matrix = mat1 + mat2
    else:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            m = cat_matrices(mat1[i], mat2[i], axis=(axis - 1))
            if m is None:
                return None
            matrix.append(m)
    return matrix
