import numpy as np
def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = linearRegCostFunction(X, y, theta, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


def linearRegCostFunction(X, y, theta, Lambda):
    """
    computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    Returns the cost and the gradient
    """
    m = len(y)

    predictions = X @ theta
    cost = 1 / (2 * m) * np.sum((predictions - y) ** 2)
    reg_cost = cost + Lambda / (2 * m) * (np.sum(theta[1:] ** 2))

    # compute the gradient

    grad1 = 1 / m * X.T @ (predictions - y)
    grad2 = 1 / m * X.T @ (predictions - y) + (Lambda / m * theta)
    grad = np.vstack((grad1[0], grad2[1:]))

    return reg_cost, grad