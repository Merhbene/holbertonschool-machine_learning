#!/usr/bin/env python3
"""
This module contains the function add_arrays
"""


def add_arrays(arr1, arr2):
    """
    Returns a new array containing the elementwise addition of arr1 and arr2,
    or None on failure
    """
    if len(arr1) != len(arr2):
        return None
    addition = []
    for i in range(len(arr1)):
        addition.append(arr1[i] + arr2[i])
    return addition
