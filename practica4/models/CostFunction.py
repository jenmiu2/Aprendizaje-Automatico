import numpy as np


def cost_function(theta1, theta2, x, y, a, lam=1):
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x
    m = X.shape[0]
    K = 10
    y_aux = np.zeros((m, K))
    cost = 0

    for i in range(1, K + 1):
        y_aux[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)

    for i in range(0, K):
        label = y_aux[:, i]
        aux = a[:, i]
        error = -label * np.log(aux) - ((1 - label) * np.log(1 - aux))
        cost = cost + sum(error)

    J = 1 / m * cost
    reg_cost = J + lam / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    return J, reg_cost
