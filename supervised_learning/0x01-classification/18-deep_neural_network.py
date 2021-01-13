#!/usr/bin/env python3

class DeepNeuralNetwork:
  def __init__(self, nx, layers):
    self.nx=nx
    self.layers=layers

    if type(self.nx)!=int:
      raise Exception("nx must be an integer")
    if self.nx<1 :
      raise Exception("nx must be a positive integer")

    if type(self.layers)!=list or len(self.layers)==0:
      raise Exception("layers must be a list of positive integers")

    self.__L=len(self.layers)
    self.__cache={}
    self.__weights={}
    
    for i in range(self.__L):
      l=self.layers[i]
      if not isinstance(l , int) or l==0 :
        raise Exception("layers must be a list of positive integers")
      if i>0:
         self.__weights["W"+str(i+1)]=np.random.randn(l,self.layers[i-1])*np.sqrt(2/self.layers[i-1])
         self.__weights["b"+str(i+1)]=np.zeros(shape=(l,1))
      if i==0 :
         self.__weights["W1"]=np.random.randn(l,self.nx)*np.sqrt(2/self.nx)
         self.weights["b1"]=np.zeros(shape=(l,1))



  @property
  def L(self):
    return(self.__L)
  @property  
  def cache(self):
    return(self.__cache)
  @property
  def weights(self):
    return(self.__weights)

  def forward_prop(self, X):
    m=X.shape[1]
    for i in range(self.__L+1):
      if i==0:
        self.__cache["A0"]=X
      else :
        z=np.dot(self.__weights["W"+str(i)],self.__cache["A"+str(i-1)])+self.__weights["b"+str(i)]
        self.__cache["A"+str(i)]=1/(1+np.exp(-z))

    return self.__cache["A"+str(self.__L)] , self.__cache
    

