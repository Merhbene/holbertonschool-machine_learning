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

  def cost(self, Y, A):
    m=Y.shape[1]
    c = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001  - A)))  # compute cost

    return c

  def evaluate(self, X, Y):
    A,cache=self.forward_prop(X)
    cost=self.cost(Y,A)
    return np.round(A),cost 


  def gradient_descent(self, Y, cache, alpha=0.05):
    m=Y.shape[1]
    dz=cache["A"+str(self.__L)]-Y
    for i in range(self.__L,0,-1):

        dw=(1/m)*np.dot(dz,cache["A"+str(i-1)].T)
        db=(1/m)*np.sum(dz,axis=1,keepdims=True)

        self.__weights["W"+str(i)]=self.weights["W"+str(i)]-(alpha*dw)
        self.__weights["b"+str(i)]=self.__weights["b"+str(i)]-(alpha*db)

        dA=cache["A"+str(i-1)]*(1-cache["A"+str(i-1)])
        dz=np.dot(self.__weights["W"+str(i)].T,dz)*dA

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
        A,self.__cache=self.forward_prop(X)
        self.gradient_descent(Y,self.__cache,alpha)


      
      return self.evaluate(X,Y)
      
      
        

