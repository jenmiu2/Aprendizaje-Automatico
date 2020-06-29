from sklearn.svm import SVC


def initial(x, y, c, sigma):
    gamma = 1 / (2 * sigma**2)
    svm = SVC(kernel='rbf', C=c, gamma= gamma)
    svm.fit(x, y)
    return svm