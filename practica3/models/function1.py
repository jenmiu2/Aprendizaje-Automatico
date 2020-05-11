import numpy as np
import scipy.optimize as opt


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def cost_function(theta, x, y):
    h = sig_function(np.matmul(x, theta))
    y_ravel = np.ravel(y)
    h[h == 1] = 0.999
    cost = (-1 / (len(x))) * (np.dot(y_ravel, np.log(h)) + np.dot((1 - y_ravel), np.log(1 - h)))
    return cost


def grad_function(theta, x, y):
    h = sig_function(np.matmul(x, theta))
    y_ravel = np.ravel(y)

    grad = (1 / len(x)) * np.matmul(x.T, h - y_ravel)
    return grad


def cost_function_reg(theta, x, y, lam=1):
    h = sig_function(np.matmul(x, theta))
    h[h == 1] = 0.999
    y_ravel = np.ravel(y)
    cost = (-1 / (len(x))) * (np.dot(y_ravel, np.log(h)) + np.dot((1 - y_ravel), np.log(1 - h)))
    cost = cost + (lam / (2 * len(x)) * np.dot(theta, theta))
    return cost


def grad_function_reg(theta, x, y, lam=1):
    h = sig_function(np.matmul(x, theta))
    y_ravel = np.ravel(y)
    grad = (1 / len(x)) * np.matmul(x.T, h - y_ravel)
    grad = grad + (lam / len(x)) * theta
    return grad


def oneVsAll(x, y, num_etiquetas, reg, theta):
    label = (y == num_etiquetas).astype(int)
    reg[num_etiquetas, :] = opt.fmin_tnc(func=cost_function, x0=theta, fprime=grad_function, args=(x, label))[1]
    return reg
