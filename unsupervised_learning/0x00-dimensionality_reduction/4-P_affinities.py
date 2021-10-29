#!/usr/bin/env python3
""" P affinities """
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):

    n = X.shape[0]
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n): # per data point
    
        low = None
        high = None

        Di = np.append(D[i, :i], D[i, i+1:]) 
        # or Di = np.delete(D[i], i, axis=0)
        (Hi, Pi) = HP(Di, betas[i]) #  the Shannon entropy and P affinitie relative to that data point

        Hdiff = Hi - H # the difference in Shannon entropy from perplexity for all Gaussian distributions

        while np.abs(Hdiff) > tol: # binary search for the Hi(bete-i/ segma-i) until the difference attend the maximum tolerance allowed

            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + low) / 2

            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H

        P[i, :i] = Pi[:i] 
        P[i, i+1:] = Pi[i:]
        """
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi
        """
    # symmetric
    P = (P + P.T)/(2*n)
    return P
