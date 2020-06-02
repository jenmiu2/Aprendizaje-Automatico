import matplotlib.pyplot as plt
import numpy as np
from models import RegLinealReg
from models import readFile
from models import ca

x, y, xVal, yVal = readFile.image()

def parte1():
    M = len(x)
    X = np.hstack((np.ones((M, 1)), x))
    theta = np.ones(2)
    plt.scatter(x, y, marker="x", color="r")

    cost, grad = RegLinealReg.LinearGradienteCost(theta, X, y, 1)
    print("Cost: {}".format(cost))
    print("Grad 0: {}, Grad 1: {}".format(grad[0], grad[1]))

    fmin = RegLinealReg.minGradient(X, y, theta, 0)

    print("Cost: {}".format(fmin))
  #  print("Theta 0: {}, Theta 1: {}".format(theta[0], theta[1]))


    maxx, minx = int(np.amax(x)), int(np.min(x))
    print("x :: Max: {}, Min: {}".format(maxx, minx))
    x_value = [j for j in range(minx, maxx)]

    y_value = [i * theta[1] + theta[0] for i in x_value]
    maxy, miny = int(np.amax(y_value)), int(np.min(y_value))
    print("y :: Max: {}, Min: {}".format(maxy, miny))
    plt.plot(x_value, y_value, marker="x", color="b")
    plt.xlim(minx, maxx)

    plt.show()


def parte2():
    X = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
    XVal = np.hstack((np.ones(shape=(xVal.shape[0], 1)), xVal))
    errTrain, errVal = ca.learningCurve(X, y, XVal, yVal)
    print(errTrain)
    print(errVal)


parte1()
