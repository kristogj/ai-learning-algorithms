import os
import gzip
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


def load_data(path, mode='train'):
    """
    Load MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def uniform_random_selection(Xtrain, ytrain, M):
    """
    Take the Xtrain and ytrain data and pick M random samples that you return
    :param Xtrain: 60k images of digits
    :param ytrain: 60k labels to the images
    :param M: Number of random selections you want to return
    :return: M randomly selected images with their corresponding labels
    """
    # Select M random indices in the range of number of images in Xtrain
    indices = np.random.choice(Xtrain.shape[0], M, replace=False)

    return Xtrain[indices], ytrain[indices]


def get_model(Xtrain, ytrain, algorithm='auto', metric='euclidean', n_neighbors=1):
    """
    Train a KNeighborsClassifier
    :param Xtrain: Training images for the model
    :param ytrain: Labels for the images the model is "trained" on
    :return: KNeighborsClassifier
    """
    model = KNeighborsClassifier(algorithm=algorithm, metric=metric, n_neighbors=n_neighbors)
    return model.fit(Xtrain, ytrain)


def test(model, Xtest, ytest):
    """
    Test the model on test data, and return the error rate
    :param model: KNeighborsClassifier
    :param Xtest: Test images
    :param ytest: Labels for the test images
    :return: Error rate
    """
    predictions = model.predict(Xtest)
    error_rate = sum(predictions != ytest) / ytest.shape[0]
    return error_rate


def get_stats(error_rates):
    mean = np.mean(error_rates)
    std = np.std(error_rates)
    maxi = np.max(error_rates)
    mini = np.min(error_rates)
    confidence_interval = st.t.interval(0.95, len(error_rates) - 1, loc=mean, scale=st.sem(error_rates))
    print("MEAN: {} \nSTD: {} \nMAX: {} \nMIN: {} \nCONF_INT: {}".format(mean, std, maxi, mini, confidence_interval))
    return mean, std, maxi, mini, confidence_interval


if __name__ == '__main__':
    # Settings for model
    ALGORITHM = "kd_tree"  # Algorithm used to make model more efficient
    METRIC = "euclidean"  # Distance metric in the model
    N_NEIGHBORS = 1  # Number of neighbors in k-NN (Assignments ask for 1-NN, so use k=1)

    # Settings for prototyping
    M = 1000  # Number of images to sample
    N_COMPONENTS = 40  # Number of components to use for the PCA
    K = 5  # Number of folds

    # Load data
    Xtrain, ytrain = load_data("./", mode="train")
    Xtest, ytest = load_data("./", mode="t10k")

    error_rates = []  # List of error rates for each fold
    # In order to get error bars we need to do the prototyping k times
    for fold in range(1, K + 1):
        # Do prototyping
        Xtrain_rand, ytrain_rand = uniform_random_selection(Xtrain, ytrain, M)

        # To reduce dimensionality and run time we PCA transform the input
        pca = PCA(n_components=N_COMPONENTS)
        Xtrain_rand = pca.fit_transform(Xtrain_rand)
        Xtest_transformed = pca.transform(Xtest)

        # Initialize a model
        model = get_model(Xtrain_rand, ytrain_rand, algorithm=ALGORITHM, metric=METRIC, n_neighbors=N_NEIGHBORS)

        # Test model
        error_rate = test(model, Xtest_transformed, ytest)
        error_rates.append(error_rate)
        print("FOLD: {}, ERROR RATE: {}".format(fold, error_rate))

    get_stats(error_rates)
