import numpy as np
from sklearn.preprocessing import StandardScaler


def matrixH(x, p):
    for i in range(2, p + 1):
        X = np.hstack((x, (x[:, 0] ** i)[:, np.newaxis]))
    return X
