import numpy as np
import models.basicFunction as bf
import models.function1 as f1
import models.graph as graph
import models.function2 as f2
import scipy.optimize as opt

#Read Data
def parte1():

    x, y = bf.read_file()
    theta = np.array([0.0, 0.0, 0.0])

    grad = f1.grad_function(theta, x, y)
    cost = f1.cost_function(theta, x, y)

    result = opt.fmin_tnc(func=f1.cost_function, x0=theta, fprime=f1.grad_function, args=(x, y))
    theta_opt = result[0]
    porAdmin = f1.porcentaje_ej(x.dot(theta_opt))
    print("Gradiente: {} Coste: {} Tetha: {} Porcentaje Admitidos: {}%".format(grad, cost, theta_opt, porAdmin))

    #mostrar los datos
    graph.show_fronteir(x, y, theta_opt)

def parte2():
    x, y = bf.read_file_2()
    theta = np.zeros(28)
    #   graph.show_parte2(x, y)
    x = f2.map_attr(x)

    cost = f2.cost_function(theta, x, y)
    grad = f2.grad_function(theta, x, y)
    result = opt.fmin_tnc(func=f2.cost_function, x0=theta, fprime=f2.grad_function, args=(x, y))
    theta_opt = result[0]
    print("Gradiente: {} Coste: {} Tetha: {}".format(grad, cost, theta_opt))

    #mostrar datos
    graph.show_fronteir2(x, y, theta_opt)

parte2()