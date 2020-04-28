import numpy as np
import scipy.optimize as opt

def sig_function(x):
    exp = np.exp(-x)
    result = (1 / (1 + exp))
    return result


def cost_function(x, y, theta=[0.01, 0.001]):
    h = sig_function(np.matmul(x, theta))
    cost = (-1 / (len(x))) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost


def grad_function(x, y, theta=[0.01, 0.001]):
    h = sig_function(np.matmul(x, theta))
    grad = (1 / len(x)) * np.matmul(x.T, h - y)
    return grad


def opt_values(cost, grad, x, y, theta=[0.01, 0.001]):
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=grad, args=(x, y))
    theta_opt = result[0]
    return theta_opt
