import numpy as np
from scipy.optimize import minimize

jValHistory = []

def LinearGradienteCost(theta, x, y, lam=0):
    h = x @ theta
    m = len(y)

    cost = (1 / (2 * m)) * sum((h - y) ** 2)
    reg_cost = cost + lam / (2 * m) * sum(theta ** 2)


    # para j = 0
    grad = (1 / m) * (x.T @ (h - y))

    # para j >= 1
    grad[1:] = grad[1:] + ((lam / m) * theta[1:])

    jValHistory.append(reg_cost)
    return reg_cost, grad


def minGradient(x, y, theta, lam=0):
    #options={'maxiter': 200}
    #method='L-BFGS-B'
    fmin = minimize(fun=LinearGradienteCost,
                    x0=theta,
                    args=(x, y, lam),
                    method='TNC',
                    jac=True,
                    options={'maxiter': 3000})

    return fmin
