import numpy as np


def cost_function(x, y):
    m = len(y)
    K = 10
    cost = 0

    for i in range(0, K):
        label = (y == i).astype(int)
        cost += -y * np.log(sig_function(x[i])) - (1 - sig_function(x[i]))

    cost = cost * (1 / m)
    return cost


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s
