#!/usr/bin/env python3
"new class"


class Binomial():
    "Binomial distribution"
    def __init__(self, data=None, n=1, p=0.5):

        if data is None:
            if (n <= 0):
                raise ValueError("n must be a positive value")
            if (p >= 1) or (p <= 0):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")

            # Calculate the n of p
            mean = sum(data) / len(data)
            var = sum([(i - mean) ** 2 for i in data]) / len(data)
            p = 1 - (var / mean)
            self.n = round(mean / p)
            self.p = mean / self.n
