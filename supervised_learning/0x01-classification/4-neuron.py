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


  def forward_prop(self, X):
    #m=X.shape[1]
    z=np.dot(self.__W,X)+self.__b
    self.__A=(1/(1+np.exp(-z)))
    return  self.__A



  def cost(self, Y, A):
    m=Y.shape[1]
    c = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001  - A)))   # compute cost using logistic regression
    return c

  def evaluate(self, X, Y):
    z = self.forward_prop(X)
    a=np.where(self.forward_prop(X)<0.5,0,1)
    return a,self.cost(Y,z)

