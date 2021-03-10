#!/usr/bin/env python3
""" NeuralNetwork class """
import numpy as np


class NeuralNetwork:
    """A neural network with one hidden layer"""
    def __init__(self, nx, nodes):

        self.nx = nx
        self.nodes = nodes

        if type(self.nx) != int:
            raise TypeError("nx must be an integer")
        if self.nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(self.nodes) != int:
            raise TypeError("nodes must be an integer")
        if self.nodes < 1:
            raise ValueError("nodes must be a positive integer")

        """Private instance attributes"""
        self.__W1 = np.random.normal(size=(self.nodes, self.nx))
        self.__b1 = np.zeros(shape=(self.nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, self.nodes))
        self.__b2 = 0
        self.__A2 = 0

    """ Getter function for W1"""
    @property
    def W1(self):
        return(self.__W1)

    """ Getter function for W2"""
    @property
    def W2(self):
        return(self.__W2)

    """ Getter function for b1"""
    @property
    def b1(self):
        return(self.__b1)

    """ Getter function for b2"""
    @property
    def b2(self):
        return(self.__b2)

    """ Getter function for A1"""
    @property
    def A1(self):
        return(self.__A1)

    """ Getter function for A2"""
    @property
    def A2(self):
        return(self.__A2)

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        nx, m = X.shape

        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = (1 / (1 + np.exp(-z1)))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = (1 / (1 + np.exp(-z2)))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression"""
        m = Y.shape[1]
        c = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)*(np.log(1.0000001 - A)))

        return c
