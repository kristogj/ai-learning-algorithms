import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from utils import sigmoid, get_data, cross_entropy_loss, accuracy
from sklearn.metrics import log_loss
import sys
from io import StringIO
from sklearn.linear_model import SGDClassifier

EPOCHS = 1
LEARNING_RATE = 2
WEIGHT_SELECTION = 3

np.random.seed(42)


class CoordinateDescentRegression:

    def __init__(self, n_features):
        self.weights = np.random.rand(n_features, 1)

    def predict(self, X):
        a = np.dot(X, self.weights)
        return sigmoid(a)

    def gradient_descent(self, X, y, predictions, config):
        error = y - predictions
        gradient = error * X
        gradient = np.sum(gradient, axis=0)

        # Select weight to update
        ws = config[WEIGHT_SELECTION]
        weight_index = 0
        if ws == "random":
            # Select weight index at random
            weight_index = np.random.randint(0, len(self.weights))
        elif ws == "best":
            # Select weight based on which have the highest gradient
            weight_index = np.argmax(np.abs(gradient))

        self.weights[weight_index] += config[LEARNING_RATE] * gradient[weight_index]


def train(model, X, y, config):
    losses, accs = [], []
    for epoch in range(1, config[EPOCHS] + 1):
        predictions = model.predict(X)

        model.gradient_descent(X, y, predictions, config)

        loss = log_loss(y, predictions)
        acc = accuracy(y, predictions)

        losses.append(loss)
        accs.append(acc)

        if epoch % 100 == 0:
            print("Epoch {}, Loss {}, Acc {}".format(epoch, round(loss, 3), round(acc, 3)))

    return losses, accs


def sklearn_classifier(X, y, config):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Init model
    model = SGDClassifier(loss="log", alpha=0, max_iter=config[EPOCHS], verbose=1, learning_rate="constant",
                          eta0=config[LEARNING_RATE], tol=1e-10, n_iter_no_change=100)
    model.fit(X, y.reshape(-1))

    # Get loss from std
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if (len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    return model, loss_list


if __name__ == '__main__':
    config = {
        EPOCHS: 1000,
        LEARNING_RATE: 0.05
    }
    X, y = get_data()

    # Sklearn model
    model, sklearn_losses = sklearn_classifier(X, y, config)

    # Random selection coordinate descent
    print("Random Model")
    config[WEIGHT_SELECTION] = "random"
    rand_model = CoordinateDescentRegression(X.shape[1])
    rand_losses, rand_accs = train(rand_model, X, y, config)

    # config[LEARNING_RATE] = 0.00001
    # Best selection coordinate descent
    print("BEST Model")
    config[WEIGHT_SELECTION] = "best"
    best_model = CoordinateDescentRegression(X.shape[1])
    best_losses, best_accs = train(best_model, X, y, config)

    x = np.arange(1, len(best_losses) + 1)
    plt.plot(x, sklearn_losses, label="SGDClassifier")
    plt.plot(x, rand_losses, label="Random")
    plt.plot(x, best_losses, label="Best")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(loc="best")
    plt.savefig("./plot.png")
    plt.show()
