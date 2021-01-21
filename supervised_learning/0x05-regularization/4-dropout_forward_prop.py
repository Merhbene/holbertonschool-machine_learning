#!/usr/bin/env python3
"A Dropout function"
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob): 
    "conduct forward propagation using Dropout"
    m = X.shape[1]
    D={}
    for i in range(L + 1):
       if i == 0:
          D["D1"] = np.random.binomial(1, keep_prob, X.shape)
          D["A0"] = X * D["D1"]

       else:
          z = np.matmul(weights["W" + str(i)], 
                    D["A" + str(i-1)]) + weights["b" + str(i)]
          A = np.tanh(z) 
          D["D" + str(i+1)] = np.random.binomial(1, keep_prob, A.shape)
          D["A" + str(i)] = A * D["D" + str(i+1)]

    return D
