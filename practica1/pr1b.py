import numpy as np

def dev_estandar(x):
    mu = np.mean(x)
    sigma = np.std(x) #desviacion estandar
    x_norm = (x - mu) / sigma
    return mu, sigma, x_norm


def fun_coste(x, y, theta):
   h = np.dot(x, theta)
   aux = (h - y)
   aux1 = aux.transpose()
   h_trans = np.dot(aux, aux1)

   return (h_trans.sum() / (2 * len(x)))
