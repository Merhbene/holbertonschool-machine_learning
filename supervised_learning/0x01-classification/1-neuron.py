#!/usr/bin/env python3

class Neuron:
  def __init__(self,nx):
    self.nx=nx
    if type(self.nx)!=int:
      raise Exception("nx must be an integer")
    if self.nx<1 :
      raise Exception("nx must be a positive integer")
    self.__W=np.expand_dims(np.random.normal(size=self.nx),axis=0)
    self.__b=0
    self.__A=0
  @property
  def W(self):
    return(self.__W)
  @property  
  def b(self):
    return(self.__b)
  @property
  def A(self):
    return(self.__A)
    