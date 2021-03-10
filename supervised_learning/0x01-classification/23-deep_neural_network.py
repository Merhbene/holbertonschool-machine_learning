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
        """ Private instance attributes"""
        self.__L = len(self.layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if not isinstance(self.layers[i], int) or (self.layers[i] <= 0):
                raise TypeError("layers must be a list of positive integers")
            """The weights of the network should be initialized using
            the He et al. method"""
            if i > 0:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    self.layers[i], self.layers[i-1])*np.sqrt(
                    2 / self.layers[i-1])
                self.__weights["b" + str(i + 1)] = np.zeros(
                    shape=(self.layers[i], 1))
            if i == 0:
                """The first layer"""
                self.__weights["W1"] = np.random.randn(
                    self.layers[i], self.nx)*np.sqrt(2 / self.nx)
                self.__weights["b1"] = np.zeros(shape=(self.layers[i], 1))
    """L getter"""
    @property
    def L(self):
        return(self.__L)
    """cache getter"""
    @property
    def cache(self):
        return(self.__cache)
    """weights getter"""
    @property
    def weights(self):
        return(self.__weights)

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network"""
        m = X.shape[1]
        for i in range(self.__L + 1):
            if i == 0:
                self.__cache["A0"] = X
            else:
                z = np.matmul(self.__weights["W" + str(i)],
                              self.__cache["A" + str(i-1)]) + self.__weights[
                    "b" + str(i)]
                self.__cache["A" + str(i)] = 1 / (1 + np.exp(-z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression"""
        m = Y.shape[1]
        c = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)*(np.log(1.0000001 - A)))
        return c

    def evaluate(self, X, Y):
        """Evaluate the neural network’s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        A1 = np.where(A < 0.5, 0, 1)
        return A1, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        "Calculate one pass of gradient descent on the neural network"
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):

            dw = (1 / m)*np.matmul(dz, cache["A" + str(i-1)].T)
            db = (1 / m)*np.sum(dz, axis=1, keepdims=True)
            dA = cache["A" + str(i-1)]*(1 - cache["A"+str(i-1)])
            dz = np.matmul(self.__weights["W" + str(i)].T, dz) * dA
            self.__weights["W" + str(i)] = self.weights[
                "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                "b" + str(i)]-(alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if (iterations < 0):
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if (alpha < 0):
            raise ValueError("alpha must be positive")
        m = Y.shape[1]
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        Cost = []
        Iteration = []
        for i in range(iterations + 1):
            a, cost=self.evaluate(X, Y)
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if (i % step == 0 ) or (i==iterations):
              Cost.append(cost)
              Iteration.append(i)
              if (verbose) :
                 print ("Cost after", i," iterations:", cost)


      
        if (graph):
            plt.plot(Iteration,Cost)
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
