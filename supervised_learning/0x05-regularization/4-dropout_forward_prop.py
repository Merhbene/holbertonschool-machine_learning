#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
"A Dropout function"


def dropout_forward_prop(X, weights, L, keep_prob): 
    "conduct forward propagation using Dropout"
    m = X.shape[1]
    D={}
    for i in range(L + 1):
       if i == 0:
          D["A0"] = X
       else:
          z = np.matmul(weights["W" + str(i)], 
                        D["A" + str(i-1)]) + weights["b" + str(i)]
          D["A" + str(i)] = np.tanh(z) 
    
       D["D" + str(i+1)] = np.random.binomial(1, 0.8, D["A" + str(i)].shape)

    return D
