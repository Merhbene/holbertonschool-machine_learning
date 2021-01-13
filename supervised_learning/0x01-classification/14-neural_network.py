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


  def forward_prop(self, X):
    nx,m=X.shape

    #m=X.shape[1]
    z1=np.dot(self.__W1,X)+self.__b1
    self.__A1=(1/(1+np.exp(-z1)))

    z2=np.dot(self.__W2,self.__A1)+self.__b2
    self.__A2=(1/(1+np.exp(-z2)))

    return  self.__A1 , self.__A2


  def cost(self, Y, A):
    m=Y.shape[1]
    c = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001  - A)))  # compute cost

    return c

  def evaluate(self, X, Y):
    A1,A2 = self.forward_prop(X)
    a=np.where(A2<0.5,0,1)
    return(a,self.cost(Y,A2))

  def gradient_descent(self, X, Y, A1, A2, alpha=0.05):

    m=Y.shape[1]

    dz2=A2-Y
    dw2=(1/m)*np.dot(dz2,A1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)

    self.__W2=self.__W2-(alpha*dw2)
    self.__b2=self.__b2-(alpha*db2)

    dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
    dw1=(1/m)*np.dot(dz1,X.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

    self.__W1=self.__W1-(alpha*dw1)
    self.__b1=self.__b1-(alpha*db1)


    
  def train(self, X, Y, iterations=5000, alpha=0.05):

      if not isinstance(iterations, int):
        raise Exception ("iterations must be an integer")
      if (iterations<0):
        raise Exception("iterations must be a positive integer")
      if not isinstance(alpha, float):
        raise Exception ("alpha must be a float")
      if (alpha<0):
        raise Exception("alpha must be positive ")

      m=Y.shape[1]

      for i in range(iterations):
        self.__A1,self.__A2=self.forward_prop(X)
        self.gradient_descent(X,Y,self.__A1,self.__A2,alpha)
      
      return self.evaluate(X,Y)
      
      
