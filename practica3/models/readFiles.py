from scipy.io import loadmat
import numpy as np


def read_img():
    data = loadmat("data/ex3data1.mat")
    y = data['y']
    x = data['X']
    m = x.shape[0]
    x = np.hstack([np.ones([m, 1]), x])
    return x, y


def read_weight():
    data = loadmat("data/ex3data2.mat")
    theta1, theta2 = data['Theta1'], data['Theta2']
    return theta1, theta2
