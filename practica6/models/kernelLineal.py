from sklearn.svm import SVC


def initial(x, y, c):
    svm = SVC(kernel='linear', C=c)
    svm.fit(x, y)
    return svm