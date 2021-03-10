#!/usr/bin/env python3
""" pool_backward """
import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
   "perform back propagation over a pooling layer of a neural network"
   m, h_new, w_new, c_new = dA.shape
   dA_prev = np.zeros(A_prev.shape)
   kh, kw = kernel_shape[0], kernel_shape[1]
   sh, sw = stride[0], stride[1]
 
   for i in range(m):
      for h in range(h_new):
         for w in range(w_new):
            for c in range(c_new):
                A_prev_mini = A_prev[i,h*sh:h*sh+kh,w*sw:w*sw+kw,c]
                if mode == "max":
                    mask = (A_prev_mini == np.max(A_prev_mini))
                else:
                    average = 1 / (kh * kw)
                    mask = average * np.ones(kernel_shape)
                dA_prev[i, h* sh:h * sh + kh, w * sw:w * sw + kw, c] += mask * dA[i, h, w, c]
   return dA_prev     
