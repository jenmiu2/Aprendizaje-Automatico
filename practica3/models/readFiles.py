from scipy.io import loadmat
import numpy as np


def read_img():
    data = loadmat("data/ex3data1.mat")
    y = data['y']
    x = data['X']
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x
    return X, y


def read_weight():
    data = loadmat("data/ex3weights.mat")
    theta1, theta2 = data['Theta1'], data['Theta2']
    return theta1, theta2
