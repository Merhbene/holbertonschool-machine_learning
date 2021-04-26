#!/usr/bin/env python3
"new class"


class Exponential():
    "exponential distribution"
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
            self.lambtha = len(data) / sum(data)

    # Probability density function
    def pdf(self, x):
        "Calculates the value of the PDF for a given time period "
        if x < 0:
            return 0

        e = 2.7182818285

        pdf = self.lambtha * (e ** (- self.lambtha * x))
        return pdf

    # cumulative distribution function
    def cdf(self, x):
        "Calculates the value of the CDF for a given time period"

        if x < 0:
            return 0

        e = 2.7182818285
        cdf = 1 - e ** (- self.lambtha * x)
        return cdf
