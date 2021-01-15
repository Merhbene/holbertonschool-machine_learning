#!/usr/bin/env python3
""" Deep Neural Network class """
import numpy as np


class DeepNeuralNetwork:
    """A deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        self.nx = nx
        self.layers = layers
        """layers is a list representing the number of nodes in each
        layer of the network"""
        if type(self.nx) != int:
            raise TypeError("nx must be an integer")
        if self.nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(self.layers) != list or len(self.layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        """ Public instance attributes"""
        self.L = len(self.layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(self.layers[i], int) or (self.layers[i] <= 0):
                raise TypeError("layers must be a list of positive integers")
            """The weights of the network should be initialized using
            the He et al. method"""
            if i > 0:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    self.layers[i], self.layers[i-1])*np.sqrt(
                    2 / self.layers[i-1])
                self.weights["b" + str(i + 1)] = np.zeros(
                    shape=(self.layers[i], 1))
            if i == 0:
                """The first layer"""
                self.weights["W1"] = np.random.randn(
                    self.layers[i], self.nx)*np.sqrt(2 / self.nx)
                self.weights["b1"] = np.zeros(shape=(self.layers[i], 1))
