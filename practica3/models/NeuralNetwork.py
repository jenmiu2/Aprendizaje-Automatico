import numpy as np
from models import RegLogMultiClase as rl

def forwardPropagation(x, y, theta1, theta2):
    # First Input Layer: Activation a(1)
    a_1 = x
    # Second Input Layer
    z_2 = np.matmul(theta1, a_1)
    a_2 = np.add(rl.sig_function(z_2), x[0])
    # Third Input Layer
    z_3 = np.matmul(theta2, a_2)
    h = rl.sig_function(z_3)
    return h