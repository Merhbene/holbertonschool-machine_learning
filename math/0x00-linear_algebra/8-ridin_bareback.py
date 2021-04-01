#!/usr/bin/env python3
"""
This module contains the function mat_mul
"""


def mat_mul(mat1, mat2):
    if not len(mat1) or len(mat1[0]) != len(mat2) or not len(mat2) or not \
       len(mat2[0]):
        return None
    mat = []
    for i, row in enumerate(mat1):
        mat.append([])
        for k in range(len(mat2[0])):
            sum = 0
            for j in range(len(row)):
                sum += row[j] * mat2[j][k]
            mat[i].append(sum)
    return mat
