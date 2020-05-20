import numpy as np
from models import RegLogMultiClase as rl


def forwardPropagation(x, theta1, theta2):
    # First Input Layer: Activation a(1)
    a_1 = x
    # Second Input Layer
    # theta1: shape (25, 401)
    # theta2: shape (10, 26)
    # a_1: shape (5000, 401)
    # a_2: shape (5000, 25)
    a_2 = rl.sig_function(a_1 @ theta1.T)
    aux_2 = np.ones(shape=(a_2.shape[0], a_2.shape[1] + 1))
    aux_2[:, 1:] = a_2
    a_3 = rl.sig_function(aux_2 @ theta2.T)

    return np.argmax(a_3, axis=1) + 1
