from models import readFiles as rf
from models import function1 as f1
import numpy as np


def parte1():
    x, y = rf.read_img()

    np.place(y, y == 10, 0)  # sustituimos los 10 con el 0, y.shape = (5000, 0)
    num_etiquetas = 10
    reg = np.zeros(shape=(num_etiquetas, x.shape[1]))  # x.shape = (5000, 401)

    for i in range(0, num_etiquetas):
        theta = np.zeros((x.shape[1], 1))
        reg = f1.oneVsAll(x, y, i, reg, theta)
    z = np.dot(x, reg.T)
    prediction = f1.sig_function(z)
    prediction = prediction.argmax(axis=1)
    res = np.mean(prediction == y)
    print("Accuracy: {}%".format(res * 100))


parte1()
