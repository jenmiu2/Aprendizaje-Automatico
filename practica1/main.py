import numpy as np
import pr1a as a
import pr1b as b

alphas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
r_alpha = alphas[np.random.randint(0, len(alphas) - 1)]


def practica1b():
    x, y, x_data, y_data = a.read_file()
    mu, sigma, x = b.dev_estandar(x)
    print("mu: {} sigma: {}".format(mu, sigma))

    theta = a.gradient_descent_tetha(x, y, alpha=r_alpha)
    y_predict = a.predict(theta, x_data)
    
    coste = b.fun_coste(x, y, theta)
    print(coste)
    a.show_reg_lineal_1v(x_data, y_data, y_predict)


def practica1a():
    x, y, x_data, y_data = a.read_file()
    theta = a.gradient_descent(x, y)
    y_predict = a.predict(theta, x_data)
    a.show_reg_lineal_1v(x_data, y_data, y_predict)
    a.graph_3D()


#practica1a()
practica1b()