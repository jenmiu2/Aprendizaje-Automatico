import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def read_file():
    data = pd.read_csv("data/ex1data1.csv", header=None).values
    x_data = pd.read_csv("data/ex1data1.csv", header=None, usecols=[0])
    y_data = pd.read_csv("data/ex1data1.csv", header=None, usecols=[1])
    data.astype(float)
    x = data[:, :-1]
    y = data[:, -1]
    m = x.shape[0]
    x = np.hstack([np.ones([m, 1]), x])
    return x, y, x_data, y_data


def cost_fun(x, y, theta):
    h = np.dot(x, theta)
    temp = (h - y) ** 2
    return temp.sum() / (2 * len(x))


def calculate_cost_function(theta0, theta1, x, y):
    step = 0.1
    t0 = np.arange(theta0[0], theta0[1], step)
    t1 = np.arange(theta1[0], theta1[1], step)

    t0, t1 = np.meshgrid(t0, t1)
    cost = np.empty_like(t0)

    for i, j in np.ndindex(t0.shape):
        cost[i, j] = cost_fun(x, y, [t0[i, j], t1[i, j]])

    return t0, t1, cost


def gradient_descent(x, y, m=1500, alpha=0.01):
    theta = np.array([0.01, 0.01])
    for i in range(0, m):
        theta = gradiente_aux(x, y, theta, alpha)
    return np.dot(x, theta)

def gradient_descent_tetha(x, y, m=1500, alpha=0.01):
    theta = np.array([0.01, 0.01])
    for i in range(0, m):
        theta = gradiente_aux(x, y, theta, alpha)
    return theta

def gradiente_aux(x, y, theta, alpha):
    h = np.dot(x, theta)
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    aux_theta = theta

    for i in range(n):
        aux = (h - y) * x[:, i]
        aux_theta[i] -= (alpha / m) * aux.sum()
    return aux_theta


def predict(theta, x):
    for i in range(x.shape[1]):
        y_predict = (theta[0] + (theta[1] * x))
    return y_predict


def show_reg_lineal_1v(x_data, y_data, y_predict):
   # plt.scatter(x_data, y_data, marker="x", c='#b41f2d')
    plt.plot(x_data, y_predict, color="blue", linewidth=1.0, linestyle="-")
    plt.figure(figsize=(8, 6), dpi=80)
    plt.show()


def graph_3D():
    fig = plt.figure()
    ax = Axes3D(fig)

    # make data
    x, y, x_data, y_data = read_file()
    theta0 = np.array([-10, 10])
    theta1 = np.array([-1, 4])
    x, y, z = calculate_cost_function(theta0, theta1, x, y)



    # plot surface
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=1.0, antialiased=False)

    ax.set_zlim(0, 700)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.contour(x, y, z, np.logspace(-2, 3, 20), colors='blue')
    plt.show()

