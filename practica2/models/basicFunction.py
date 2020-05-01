import numpy as np
import pandas as pd

def read_file():
    data = pd.read_csv("data/ex2data1.csv", header=None).values
    data.astype(float)
    x = data[:, :-1] #todos los datos menos la ultima columna
    y = data[:, -1] #utlima columna
    m = x.shape[0]
    x = np.hstack([np.ones([m, 1]), x])
    return x, y

def read_file_2():
    data = pd.read_csv("data/ex2data2.csv", header=None).values
    data.astype(float)
    x = data[:, :-1] #todos los datos menos la ultima columna
    y = data[:, -1] #utlima columna
    return x, y