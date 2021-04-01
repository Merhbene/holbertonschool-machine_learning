#!/usr/bin/env python3
"""
This module contains the function cat_matrices2D
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates mat1 and mat2 along a specific axis"""
    if axis > 1:
        return None
    elif axis == 1:
        if len(mat1) != len(mat2) or not len(mat1):
            return None
        mat = []
        for i in range(len(mat1)):
            mat.append([])
            mat[i] = mat1[i] + mat2[i]
    else:
        if not len(mat1) or not len(mat2) or len(mat1[0]) != len(mat2[0]):
            return None
        mat = []
        for row in mat1:
            mat.append(row)
        for row in mat2:
            mat.append(row)
    return mat
