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
        pred = f1.oneVsAll(x, y, i, reg, theta)

    for j in range(0, num_etiquetas):
        print(pred[j])
        print('num: {} -> Training Set Accuracy: {:f}'.format(j, (np.mean(pred[j] == y) * 100)))



parte1()
