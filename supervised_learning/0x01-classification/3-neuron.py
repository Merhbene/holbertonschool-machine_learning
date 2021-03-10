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

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.__b
        """ A the neuron's predection """
        self.__A = (1 / (1 + np.exp(-z)))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression"""
        m = Y.shape[1]
        c = -(1 / m)*np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        """Use 1.0000001 - A to avoid division by zero errors"""
        return c
