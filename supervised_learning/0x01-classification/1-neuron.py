#!/usr/bin/env python3
""" new class """
import numpy as np


class Neuron():
    """ Neuron class"""
    def __init__(self, nx):
        """ init function """
        self.nx = nx
        if type(self.nx) != int:
            raise TypeError("nx must be an integer")
        if self.nx < 1:
            raise ValueError("nx must be a positive integer")
        # Private instance attributes
        self.__W = np.expand_dims(np.random.normal(size=self.nx), axis=0)
        self.__b = 0
        self.__A = 0

    """Getter for W """
    @property
    def W(self):
        return(self.__W)

    """Getter for b """
    @property
    def b(self):
        return(self.__b)

    """Getter for A """
    @property
    def A(self):
        return(self.__A)
