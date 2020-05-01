import numpy as np


def sig_function(x):
    exp = np.exp(-x)
    result = (1 / (1 + exp))
    return result


def cost_function(theta, x, y):
    h = sig_function(np.matmul(x, theta))
    cost = (-1 / (len(x))) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost


def grad_function(theta, x, y):
    h = sig_function(np.matmul(x, theta))
    grad = (1 / len(x)) * np.matmul(x.T, h - y)
    return grad


def porcentaje_ej(theta):
    tam = theta.shape[0]
    sum = np.sum(theta >= 0.5)

    return 100 * sum / tam
