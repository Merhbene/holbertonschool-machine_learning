  #!/usr/bin/env python3
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
        self.__W = np.expand_dims(np.random.normal(size = self.nx), axis = 0)
        self.__b = 0
        self.__A = 0
        
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
    
