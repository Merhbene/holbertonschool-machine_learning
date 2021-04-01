#!/usr/bin/env python3
"""
This module contains the function matrix_transpose
"""


def matrix_transpose(matrix):
    T = []
    for j in range (len(matrix[0])):
        c = []
        for i in range (len(matrix)):
            c.append(matrix[i][j])
        T.append(c) 
    return T
