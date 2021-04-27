#!/usr/bin/env python3
"new class"


class Normal():
    " normal distribution"
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if (len(data) < 2):
                raise ValueError("data must contain multiple values")
            # Calculate the mean and standard deviation of data
            self.mean = sum(data) / len(data)
            L = [(i - self.mean) ** 2 for i in data]
            self.stddev = (sum(L) / len(L)) ** (1 / 2)

    def z_score(self, x):
        "Calculates the z-score of a given x-value"
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        "Calculates the x-value of a given z-score"
        x = z * self.stddev + self.mean
        return x

    def pdf(self, x):
        "Calculates the value of the PDF for a given x-value"
        pi = 3.1415926536
        e = 2.7182818285
        c = self.stddev * ((2*pi) ** (1/2))
        pdf = (e ** ( - (self.z_score(x) ** 2) / 2)) / c
        return pdf

    def erf(self, x):
        "error function"
        pi = 3.1415926536
        c = (x - (x ** 3) / 3 + (x ** 5) / 10 - (x ** 7) / 42 + (x ** 9) / 216)
        return 2 * c / (pi ** (1 / 2))

    # cumulative distribution function
    def cdf(self, x):
        "Calculates the value of the CDF for a given x-value"
        c = self.z_score(x) / (2 ** (1 / 2))
        cdf = (1 + self.erf(c)) / 2
        return cdf
