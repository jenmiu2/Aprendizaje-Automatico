import numpy as np
import scipy.optimize as opt


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


def cost_function_reg(theta, x, y, lam=0.1):
    h = sig_function(np.matmul(x, theta))
    h[h == 1] = 0.999
    y_ravel = np.ravel(y)
    cost = (-1 / (len(x))) * (np.dot(y_ravel, np.log(h)) + np.dot((1 - y_ravel), np.log(1 - h)))
    cost = cost + (lam / (2 * len(x)) * np.dot(theta, theta))
    return cost


def grad_function_reg(theta, x, y, lam=0.1):
    h = sig_function(np.matmul(x, theta))
    y_ravel = np.ravel(y)
    grad = (1 / len(x)) * np.matmul(x.T, h - y_ravel)
    grad = grad + (lam / len(x)) * theta
    return grad


def oneVsAll(x, y, num_etiquetas, reg, theta):
    label = (y == num_etiquetas).astype(int) #label.shape = (500, )
    reg[num_etiquetas, :] = opt.fmin_tnc(func=cost_function_reg, x0=theta, fprime=grad_function_reg, args=(x, label))[1]
    return reg
