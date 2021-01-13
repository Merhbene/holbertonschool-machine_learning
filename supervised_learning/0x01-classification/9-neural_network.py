#!/usr/bin/env python3

class NeuralNetwork:
  def __init__(self, nx, nodes):
    self.nx=nx
    self.nodes=nodes

    if type(self.nx)!=int:
      raise Exception("nx must be an integer")
    if self.nx<1 :
      raise Exception("nx must be a positive integer")

    if type(self.nodes)!=int:
      raise Exception("nx must be an integer")
    if self.nodes<1 :
      raise Exception("nx must be a positive integer")

    self.__W1=np.random.normal(size=(self.nodes, self.nx))
    self.__b1=np.zeros(shape=(self.nodes,1))
    self.__A1=0
    self.__W2=np.random.normal(size=(1,self.nodes))
    self.__b2=0
    self.__A2=0


  @property
  def W1(self):
    return(self.__W1)

  @property
  def W2(self):
    return(self.__W2)

  @property  
  def b1(self):
    return(self.__b1)


  @property  
  def b2(self):
    return(self.__b2)


  @property
  def A1(self):
    return(self.__A1)


  @property
  def A2(self):
    return(self.__A2)
    
    