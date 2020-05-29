from scipy.io import loadmat

# y es de dimension 5000 x 1
# x es de dimension 5000 x 400
def read_img():
    data = loadmat("data/ex5data1.mat")
    y = data['yval']
    x = data['Xval']
    return x, y

# y es de dimension 5000 x 1
# x es de dimension 5000 x 400
def read_img_test():
    data = loadmat("data/ex5data1.mat")
    y = data['ytest']
    x = data['Xtest']
    return x, y