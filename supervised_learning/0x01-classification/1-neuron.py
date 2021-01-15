  
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
        # Insert a new axis
        self.W = np.random.randn(1, self.nx)
        self.b = 0
        self.A = 0
        
    #Private instance attributes
    @property
    def W(self):
        return(self.__W)
   
    @property  
    def b(self):
        return(self.__b)
   
    @property
    def A(self):
        return(self.__A)
    
