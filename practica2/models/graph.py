
import numpy as np
from matplotlib import pyplot as plt
import models.function1 as f1
import models.function2 as f2

def show_parte1(x, y):
    isAccepted = np.where(y == 1)
    notAccepted = np.where(y == 0)

    plt.scatter(x[isAccepted, 0], x[isAccepted, 1], marker="x", c='k', label='accepted')
    plt.scatter(x[notAccepted, 0], x[notAccepted, 1], marker="o", c='#e6ac00', label='not accepted')

    plt.xlabel('Examn 1 Score')  # Add an x-label to the axes.
    plt.ylabel('Examn 2 Score')  # Add a y-label to the axes.
    plt.legend()  # Add a legend.

    plt.figure(figsize=(8, 6), dpi=80)
    plt.show()


def show_fronteir(x, y, theta):
    isAccepted = np.where(y == 1)
    notAccepted = np.where(y == 0)

    plt.scatter(x[isAccepted, 1], x[isAccepted, 2], marker="x", c='k', label='accepted')
    plt.scatter(x[notAccepted, 1], x[notAccepted, 2], marker="o", c='#e6ac00', label='not accepted')

    plt.xlabel('Examn 1 Score')  # Add an x-label to the axes.
    plt.ylabel('Examn 2 Score')  # Add a y-label to the axes.
    plt.legend()  # Add a legend.

    plt.figure()
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
    x2_min, x2_max = x[:, 2].min(), x[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = f1.sig_function(np.c_[np.ones((xx1.ravel().shape[0], 1)),
                             xx1.ravel(),
                             xx2.ravel()]
                       .dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    plt.show()


def show_parte2(x, y):
    passControl = np.where(y == 1)
    notpassControl = np.where(y == 0)

    plt.scatter(x[passControl, 1], x[passControl, 2], marker="x", c='k', label='pass')
    plt.scatter(x[notpassControl, 1], x[notpassControl, 2], marker="o", c='#e6ac00', label='not pass')

    plt.xlabel('Examn 1 Score')  # Add an x-label to the axes.
    plt.ylabel('Examn 2 Score')  # Add a y-label to the axes.
    plt.legend()  # Add a legend.

    plt.figure(figsize=(8, 6), dpi=80)
    plt.show()

def show_fronteir2(x, y, theta):
    passControl = np.where(y == 1)
    notpassControl = np.where(y == 0)

    plt.scatter(x[passControl, 1], x[passControl, 2], marker="x", c='k', label='pass')
    plt.scatter(x[notpassControl, 1], x[notpassControl, 2], marker="o", c='#e6ac00', label='not pass')

    plt.xlabel('Micro 1 Score')  # Add an x-label to the axes.
    plt.ylabel('Micro 2 Score')  # Add a y-label to the axes.
    plt.legend()  # Add a legend.

    plt.figure()
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
    x2_min, x2_max = x[:, 2].min(), x[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = f1.sig_function(f2.map_attr(np.c_[xx1.ravel(),
                                         xx2.ravel()])
                        .dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    plt.show()