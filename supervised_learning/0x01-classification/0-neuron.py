#!/usr/bin/env python3

import numpy as np


class Neuron():
    def __init__(self, nx):
       self.nx = nx
       if type(self.nx)!=int :
          raise TypeError ("nx must be an integer")
       if self.nx<1 :
          raise ValueError ("nx must be a positive integer")
       self.W = np.expand_dims(np.random.randn(self.nx), axis=0) #Insert a new axis
       self.b = 0
       self.A = 0
