from models import readFiles as rf
import numpy as np
from models import RegLogMultiClase as rl
from models import NeuralNetwork as nn


def practica3():
    x, y = rf.read_img()
    theta1, theta2 = rf.read_weight()
    np.place(y, y == 10, 0)  # sustituimos los 10 con el 0, y.shape = (5000, 0)
    num_etiquetas = 10
    reg = np.zeros(shape=(num_etiquetas, x.shape[1]))  # x.shape = (5000, 401)

    for i in range(0, num_etiquetas):
        theta = np.zeros((x.shape[1], 1)) #theta.shape = (401,)
        reg = rl.oneVsAll(x, y, i, reg, theta)

    prediction = rl.predOneVsAll(reg, x)
    sumPrediction = sum(prediction[:,np.newaxis] == y)[0] / 5000 * 100
    print("Accuracy Regression Log Multiple Clase: {}%".format(sumPrediction))

    prediction_fp = nn.forwardPropagation(x, theta1, theta2)
    sumPrediction = sum(prediction_fp[:,np.newaxis] == y) / prediction_fp.shape[0] * 100
    print("Accuracy Forward Propagation: {}%".format(sumPrediction))


practica3()
