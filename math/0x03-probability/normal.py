#!/usr/bin/env python3
"new class"


class Normal():
    def __init__(self, data=None, mean=0., stddev=1.):
        " normal distribution"

        if data is None:
            if stddev >= 0:
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
            L = [(i - self.mean) ** 2 for i in data ]
            self.stddev = (sum(L) / len(L)) ** (1 / 2)
