# !/usr/bin/env python3
"new class"


class Poisson():
    "poisson distribution"
    def __init__(self, data=None, lambtha=1.):

        if data is None:
            self.lambtha = lambtha
            if (lambtha<=0):
                raise ValueError("lambtha must be a positive value")
         
        if data is not None:
            #Calculate the lambtha of data
            if type(data) is not list:
                raise TypeError("data must be a list")
            if (len(data)<2):
                raise ValueError("data must contain multiple values")   
            L = 0
            for i in data:
                L = L + i
            self.lambtha = L / len(data)
