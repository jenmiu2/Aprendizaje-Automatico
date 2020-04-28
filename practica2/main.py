import numpy as np
import pandas as pd
import function as f
import graph as g
from matplotlib import pyplot as plt

data = pd.read_csv("data/ex2data1.csv", header=None).values
x_data = pd.read_csv("data/ex2data1.csv", header=None, usecols=[0])
y_data = pd.read_csv("data/ex2data1.csv", header=None, usecols=[1])
data.astype(float)
x = data[:, :-2]
y = data[:, -2]
m = x.shape[0]
x = np.hstack([np.ones([m, 1]), x])

x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
theta = np.array([0.01, 0.001])

plt.figure()

xx1, xx2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
h = f.sig_function(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
h.reshape(xx1.shape)

plt.countour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plt.show()

cost = f.cost_function(x, y)
grad = f.grad_function(x, y)
cost_opt = f.opt_values(cost,grad, x,y)

print("cost = {}, grad = {}, cost_opt: {}".format(cost, grad, cost_opt))

g.show_sig_data(data, cost_opt)

#mostrar los datos



