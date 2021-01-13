#!/usr/bin/env python3

def one_hot_decode(one_hot):
   classes, m=one_hot.shape
   Y_decoded=np.zeros(shape=(m,))

   for i in range(m):
     for j in range(classes):
       if one_hot[j][i]==1 :
         Y_decoded[i]=j
   return Y_decoded.astype(int)
   
