import numpy as np
from scipy.optimize import minimize

costHistory = []


def LinearGradienteCost(theta, x, y, lam=1):
    m = len(x)
    h = theta @ x.T

    # cost
    cost = 1 / 2 / m * np.sum((h - y) ** 2)
    reg_cost = cost + lam / 2 / m * np.sum(theta[1:] ** 2)

    # grad,  j = 0
    grad = (1 / m) * ((h - y).T @ x)

    # grad, j >= 1
    grad[1:] = grad[1:] + ((lam / m) * theta[1:])

    # update cost history

    costHistory.append(reg_cost)
    return reg_cost, grad


def error(x, y, theta):
    m = len(x)
    h = theta @ x.T
    err = 1 / 2 / m * np.sum((h - y) ** 2)
    return err


def minGradient(x, y, theta, lam=1):
    fmin = minimize(fun=LinearGradienteCost,
                    x0=theta,
                    args=(x, y, lam),
                    method='TNC',
                    jac=True,
                    options={'maxiter': 200})

    return fmin
