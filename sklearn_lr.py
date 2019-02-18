import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def use_function(X, Y, test_X):
    clf = LogisticRegression()
    Z = clf.fit(X.T, Y.T.astype(int))
    lr_predictions = clf.predict(test_X.T)
    print(lr_predictions)
    return lr_predictions


def test():
    X, Y = datasets.load_iris(return_X_y=True)
    use_function(X, Y)


if __name__ == "__main__":
    # test()
    a = [1, 2, 3]
    b = np.array(a).astype(np.int32)
    for i in b:
        print(int(i))
