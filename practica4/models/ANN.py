import numpy as np
from scipy.optimize import minimize


# debe funcionar para cualquier numero de entramientos y de etiquetas

# Calculate ANN with 4 Label
def backprop(theta, inputSize, hiddenSize, numLabel, x, y, reg):
    theta1 = np.reshape(theta[:inputSize * hiddenSize - 2], (numLabel, numLabel + 1))
    theta2 = np.reshape(theta[inputSize: inputSize * hiddenSize - 1], (numLabel, numLabel + 1))
    theta3 = np.reshape(theta[inputSize * hiddenSize:], (1, numLabel + 1))

    a = forwardPropagation(x, theta)
    # Calculate Gradient
    # Calculate Cost

    thetaVec = np.concatenate((np.ravel(theta1),
                               np.ravel(theta2),
                               np.ravel(theta3)))
    return thetaVec


def backwardPropagationMin(theta, inputSize, hiddenSize, num_labels, x, y, reg):
    fmin = minimize(fun=backprop,
                    x0=theta,
                    args=(inputSize, hiddenSize, num_labels, x, y, reg),
                    method='TNC',
                    jac=True,
                    options={'maxiter': 70})
    return fmin


def forwardPropagation(x, theta):
    return 0





def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s
