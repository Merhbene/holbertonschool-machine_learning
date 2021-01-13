#!/usr/bin/env python3

def one_hot_encode(Y, classes):
  m=Y.shape[0]
  Y_one_hot=np.zeros(shape=(classes, m))
  for i in range(m) :
    Y_one_hot[Y[i]][i]=1

  return Y_one_hot
  



