import numpy as np
from scipy.optimize import minimize
from models import CostFunction as cf
import math as mt


# debe funcionar para cualquier numero de entramientos y de etiquetas

def gradient(params_ns, inputSize, hiddenSize, numLabel, x, y, lam=1):
    theta1 = params_ns[:((inputSize + 1) * hiddenSize)].reshape(hiddenSize, inputSize + 1)
    theta2 = params_ns[((inputSize + 1) * hiddenSize):].reshape(numLabel, hiddenSize + 1)

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x

    m = x.shape[0]

    a1, z2, a2, a3, h = forwardPropagation(x, theta1, theta2)
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))  # (1, 26)
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = 1 / m * delta1
    delta2 = 1 / m * delta2

    delta1Reg = delta1 + (lam / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    delta2Reg = delta2 + (lam / m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

    # Calculate Cost
    jVal, jValGrad = cf.cost_function(theta1=theta1, theta2=theta2, x=x, y=y, a=h, numLabel=numLabel)

    deltaVec = np.concatenate((delta1Reg.ravel(), delta2Reg.ravel()))
    return jValGrad, deltaVec


def forwardPropagation(x, theta1, theta2):
    # First Input Layer: Activation a(1)
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x
    a1 = X
    # Second Input Layer
    # theta1: shape (25, 401)
    # a1: shape (5000, 401)
    # a2: shape (5000, 25)
    z2 = a1 @ theta1.T
    a2 = sig_function(z2)

    # aux2: shape (5000, 26) adding one column of one´s to a2
    aux2 = np.ones(shape=(a2.shape[0], a2.shape[1] + 1))
    aux2[:, 1:] = a2

    # Third Input Layer
    # theta2: shape (10, 26)
    # a3: shape (5000, 26)
    a3 = aux2 @ theta2.T
    h = sig_function(a3)

    return a1, z2, aux2, a3, h


def sig_dev_function(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def randomWeight(Lin, Lout, e=0.12):
    e = mt.sqrt(6) / mt.sqrt(Lin + Lout)
    matrix = np.random.random(size=(Lout, Lin + 1))
    matrix = matrix * 2 * e
    matrix = matrix - e
    return matrix


def backpropagationLearning(x, y, theta1, theta2, hiddenSize=25, numLabel=10, inputSize=400):
    # Initialize params
  #  params_ns = (np.random.random(size=hiddenSize * (inputSize + 1) +
       #                                numLabel * (hiddenSize + 1)) - 0.5) * 0.25
    params_ns = np.concatenate((theta1.ravel(), theta2.ravel()))

    fmin = minimize(fun=gradient,
                    x0=params_ns,
                    args=(inputSize, hiddenSize, numLabel, x, y, 1),
                    method='TNC',
                    jac=True,
                    options={'maxiter': 70})

    return fmin
