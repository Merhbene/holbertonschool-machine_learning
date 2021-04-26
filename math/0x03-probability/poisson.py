#!/usr/bin/env python3
"new class"


class Poisson():
    "poisson distribution"
    def __init__(self, data=None, lambtha=1.):

        if data is None:
            self.lambtha = float(lambtha)
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")

            # Calculate the lambtha of data
            self.lambtha = sum(data) / len(data)

    # probability mass function
    def pmf(self, k):
        "Calculates the value of the PMF for a given number of “successes” "
        k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i
        P = ((self.lambtha ** k) * (e ** (- self.lambtha))) / fact_k
        return P

    # cumulative distribution function
    def cdf(self, k):
        "Calculates the value of the CDF for a given number of “successes” "
        k = int(k)
        if k < 0:
            return 0

        e = 2.7182818285
        cdf = 0
        for i in range(k+1):

            fact_i = 1
            for j in range(1, i + 1):
                fact_i *= j

            cdf += ((self.lambtha ** i) * (e ** (- self.lambtha))) / fact_i
        return cdf
