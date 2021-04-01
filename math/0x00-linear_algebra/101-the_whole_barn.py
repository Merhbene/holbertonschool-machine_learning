#!/usr/bin/env python3


def add_matrices(mat1, mat2):

    matrix = []

    if type(mat1) is not type(mat2) or type(mat1) is not list or \
       len(mat1) != len(mat2) or not len(mat1):
        return None
    if type(mat1[0]) is list:
        for i in range(len(mat1)):
            m = add_matrices(mat1[i], mat2[i])
            if m is None:
                return None
            matrix.append(m)
    else:
        for i in range(len(mat1)):
            matrix.append(mat1[i] + mat2[i])
    return matrix
