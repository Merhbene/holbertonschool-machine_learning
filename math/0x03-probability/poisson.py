#!/usr/bin/env python3
"new class"


def factorial(x):
    f = 1
    for i in range(1, x + 1):
        f *= i
    return f

class Poisson():
    "poisson distribution"
    def __init__(self, data=None, lambtha=1.):

        if data is None:
            self.lambtha = float (lambtha)
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")

        if data is not None:
            # Calculate the lambtha of data
            if type(data) is not list:
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")
            L = 0
            for i in data:
                L = L + i
            self.lambtha = L / len(data)

    # probability mass function
    def pmf(self, k):
        "Calculates the value of the PMF for a given number of “successes” "
        k = int (k)
        if k < 0:
            return 0
        e = 2.7182818285
        P = ((self.lambtha ** k) * (e ** (- self.lambtha))) / factorial(k)
        return P
