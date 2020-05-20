from scipy.io import loadmat
import numpy as np

# y es de dimension 5000 x 1
# x es de dimension 5000 x 400
def read_img():
    data = loadmat("data/ex4data1.mat")
    y = data['y']
    x = data['X']
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x
    return X, y

# Theta1 es de dimensión 25 x 401
# Theta2 es de dimensión 10 x 26
def read_weight():
    data = loadmat("data/ex4weights.mat")
    theta1, theta2 = data['Theta1'], data['Theta2']
    return theta1, theta2
