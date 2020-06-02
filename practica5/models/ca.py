from models import RegLinealReg
import numpy as np


def learningCurve(x, y, xVal, yVal, lam=0):
    m = len(y)
    errTrain, errVal = [], []

    for i in range(1, m + 1):
        cost, theta = RegLinealReg.calculateError(np.zeros(shape=(2, 1)), x[0: i], y[0: i], 0)
        cost = RegLinealReg.calculateError(theta,  x[0: i, :], y[0: i, :])[0]
        errTrain.append(cost)
        costVal = RegLinealReg.calculateError(theta,  xVal[0: i, :], yVal[0: i, :])[0]
        errVal.append(costVal)
    return errTrain, errVal
