#!/usr/bin/env python3
"Moving Average"


def moving_average(data, beta):
    "calculates the weighted moving average of a data set"
    v = 0
    Avg = []
    for i, d in enumerate(data):
        v = beta * v + (1 - beta) * d
        bias_corr = v / (1 - (beta ** (i + 1)))
        Avg.append(bias_corr)
    return Avg
