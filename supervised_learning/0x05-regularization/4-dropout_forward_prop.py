
#!/usr/bin/env python3
"A Dropout function"
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob): 
    "conduct forward propagation using Dropout"
    m = X.shape[1]
    D={}
    D["A0"] = X 

    for i in range(1, L):
       z = np.matmul(weights["W" + str(i)], D["A" + str(i-1)]) + weights["b" + str(i)]

       if i == L-1 :
          A = (np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True))#softmax
          D["A" + str(i+1)] = A
       else:

          A = np.tanh(z) 
          D["D" + str(i)] = np.random.binomial(1, keep_prob, A.shape)
          D["A" + str(i)] = (A * D["D" + str(i)]) / keep_prob 

    return D
