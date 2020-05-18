from models import readFiles as rf
import numpy as np
from models import RegLogMultiClase as rl


def parte1():
    x, y = rf.read_img()

    np.place(y, y == 10, 0)  # sustituimos los 10 con el 0, y.shape = (5000, 0)
    num_etiquetas = 10
    reg = np.zeros(shape=(num_etiquetas, x.shape[1]))  # x.shape = (5000, 401)

    for i in range(0, num_etiquetas):
        theta = np.zeros((x.shape[1], 1)) #theta.shape = (401,)
        reg = rl.oneVsAll(x, y, i, reg, theta)

    # prediction = rl.predOneVsAll(reg, x)
    # sumPrediction = sum(prediction[:,np.newaxis] == y)[0] / 5000 * 100
    # print("Accuracy: {}%".format(sumPrediction))


parte1()
