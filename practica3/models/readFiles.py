from scipy.io import loadmat

def read_part1():
    data = loadmat("data/ex3data1.mat")
    print(data.keys())
    y = data['y']
    x = data['X']
    return x, y