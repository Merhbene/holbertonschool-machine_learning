#!/usr/bin/env python3
"""
This module contains the function add_matrices2D
"""


def add_matrices2D(mat1, mat2):
    """returns the elementwise addition of two 2D matrices"""
    if len(mat1) != len(mat2) or not len(mat1) or len(mat1[0]) != len(mat2[0]):
        return None
    addition = []
    for i in range(len(mat1)):
        addition.append([])
        for j in range(len(mat1[i])):
            addition[i].append(mat1[i][j] + mat2[i][j])
    return addition
