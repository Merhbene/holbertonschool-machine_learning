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

    def evaluate(self, X, Y):
        """Evaluate the neuron’s predictions"""
        z = self.forward_prop(X)
        a = np.where(self.forward_prop(X) < 0.5, 0, 1)
        return a, self.cost(Y, z)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        """ ==> dz shape =(1,m)"""
        dw = (1 / m) * (np.matmul(dz, X.T))
        """ shape (1,m) * (m,nx) ==> dw shape = (1,nx)"""
        db = (1 / m) * np.sum(A - Y)
        """ Update weights and bias """
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if (iterations < 0):
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if (alpha < 0):
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        m = Y.shape[1]
        Cost = []
        Iteration = []
        for i in range(iterations):
            a, cost = self.evaluate(X, Y)
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

            if (i % step == 0 ):
                Cost.append(cost)
                Iteration.append(i)
                if (verbose):
                    print("Cost after", i," iterations:", cost)  
        if (graph):
             plt.plot(Iteration, Cost, 'b')
             plt.ylabel('cost')
             plt.xlabel('iteration')
             plt.title("Training Cost")
             plt.show()

        return self.evaluate(X, Y)
