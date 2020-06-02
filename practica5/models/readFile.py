from scipy.io import loadmat
import numpy as np

# y es de dimension 12 x 1
# x es de dimension 12 x 1
# x_val de dimension 21 x 1
# y_val de dimension 21 x 1
def image():
    data = loadmat("data/ex5data1.mat")
    X = data["X"]
    y = data["y"]
    yVal = data['yval']
    xVal = data['Xval']
    return X, y, xVal, yVal

# y es de dimension 5000 x 1
# x es de dimension 5000 x 400
def image_Test():
    data = loadmat("data/ex5data1.mat")
    y = data['ytest']
    x = data['Xtest']
    return x, y