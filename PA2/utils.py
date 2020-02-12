import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize


def cross_entropy_loss(y, predictions):
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    return - np.sum(y * np.log(predictions)) / y.shape[0]


def accuracy(y, predictions):
    predictions = np.around(predictions)
    predictions = predictions.reshape(-1)
    y = y.reshape(-1)
    return sum(predictions == y) / y.shape[0]


def get_data():
    data = genfromtxt("./wine.data", delimiter=",")
    np.random.shuffle(data)
    X, y = data[:, 1:], data[:, 0]
    X = normalize(X)

    # Add bias to first column
    b = np.ones((X.shape[0], X.shape[1] + 1))
    b[:, 1:] = X
    X = b
    y = y.astype("int").reshape((-1, 1)) - 1
    return X, y


def sigmoid(a):
    return 1 / (1 + np.exp(-a))
