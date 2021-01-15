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

        """Public instance attributes"""
        self.W1 = np.random.normal(size=(self.nodes, self.nx))
        self.b1 = np.zeros(shape=(self.nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, self.nodes))
        self.b2 = 0
        self.A2 = 0
