#!/usr/bin/env python3
"""
This module contains the function matrix_shape
"""


def matrix_shape(matrix):
    """Calculates the shape of matrix"""
    shape = []
    while type(matrix) is list and len(matrix) > 0:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
