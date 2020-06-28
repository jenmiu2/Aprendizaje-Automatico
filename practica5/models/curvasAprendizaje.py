from models import RegLinealReg
import numpy as np


def learningCurve(x, y, theta, xVal, yVal):
    m = len(x)
    errTrain, errVal = np.zeros(m), np.zeros(m)

    for i in range(1, m + 1):
        # train values
        x_c = x[:i]
        y_c = y[:i]

        # minimize gradient
        res = RegLinealReg.minGradient(theta, x_c, y_c)

        # calculate the error
        errTrain[i - 1] = RegLinealReg.error(res['x'], x_c, y_c)
        errVal[i - 1] = RegLinealReg.error(res['x'], xVal, yVal)

    return errTrain, errVal
