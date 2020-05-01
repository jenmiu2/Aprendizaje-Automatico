import numpy as np
import models.function1 as f1
from sklearn.preprocessing import PolynomialFeatures


def map_attr(x):
    poly = PolynomialFeatures(6)
    x = poly.fit_transform(x)
    return x


def cost_function(theta, x, y, lam=1):
    h = f1.sig_function(np.matmul(x, theta))
    cost = (-1 / (len(x))) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    cost = cost + (lam / (2 * len(x)) * np.dot(theta, theta))
    return cost


def grad_function(theta, x, y, lam=1):
    h = f1.sig_function(np.matmul(x, theta))
    grad = (1 / len(x)) * np.matmul(x.T, h - y)
    grad = grad + (lam / len(x)) * theta
    return grad
