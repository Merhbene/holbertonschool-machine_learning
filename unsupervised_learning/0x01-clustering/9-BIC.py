
   
#!/usr/bin/env python3
""" Defines `BIC`. """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """calculates BIC over various """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if type(kmin) is not int or kmin != int(kmin) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n 
    if type(kmax) is not int or kmax != int(kmax) or kmax < 1:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations != int(iterations) or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    results = []
    log_likelihoods = []
    BICs = []
    for cluster_count in range(kmin, kmax + 1):
        priors, centroids, covariances, responsibilities, log_likelihood = \
            expectation_maximization(
                X, cluster_count, iterations, tol, verbose)
        results.append((priors, centroids, covariances))
        log_likelihoods.append(log_likelihood)
        parameter_count = (
            cluster_count * (dimention_count + 2) * (dimention_count + 1) / 2
            - 1
        )
        BICs.append(
             np.log(sample_count) * parameter_count - 2 * log_likelihood)

    best_index = np.argmin(BICs)
    best_cluster_count = kmin + best_index
    best_parameters = results[best_index]

    return (
        best_cluster_count,
        best_parameters,
        np.array(log_likelihoods),
        np.array(BICs)
    )
