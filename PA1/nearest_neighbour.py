import os
import gzip
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import csv

ALGORITHM = "ALGORITHM"
METRIC = "METRIC"
N_COMPONENTS = "N_COMPONENTS"
K = "K"
M = "M"
SAMPLER = "SAMPLER"


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


def k_means_sampler(Xtrain, ytrain, M):
    """
        Use K-means for every digit, and find N = M // n_classes centroids.
        Return these centroids as samples for that digit.
        :param Xtrain: 60k images of digits
        :param ytrain: 60k labels to the images
        :param M: Number of samples you want to return
        :return: M selected images with their corresponding labels
        """
    # Sort images pr label
    sorted_images = defaultdict(list)
    for image, label in zip(Xtrain, ytrain):
        sorted_images[label].append(image)

    N = M // 10

    X, y = [], []
    # Do K-means pr label, and return n_images pr label
    for label in range(10):
        data = np.array(sorted_images[label])

        # PCA transform before k-means to make clustering go faster
        pca = PCA(n_components=10)
        data_transformed = pca.fit_transform(data)

        # Do k-means and find N number of centroids
        kmeans = KMeans(n_clusters=N).fit(data_transformed)
        centroids = pca.inverse_transform(kmeans.cluster_centers_)
        X += centroids.tolist()
        y += [label] * N

    return np.array(X), np.array(y)


def get_model(Xtrain, ytrain, algorithm='auto', metric='euclidean', n_neighbors=1):
    """
    Train a KNeighborsClassifier
    :param Xtrain: Training images for the model
    :param ytrain: Labels for the images the model is "trained" on
    :return: KNeighborsClassifier
    """
    model = KNeighborsClassifier(algorithm=algorithm, metric=metric, n_neighbors=n_neighbors)
    return model.fit(Xtrain, ytrain)


def test(Xtrain, ytrain, Xtest, ytest, config):
    """
    Test the model on test data, and return the error rate
    :param model: KNeighborsClassifier
    :param Xtest: Test images
    :param ytest: Labels for the test images
    :return: Error rate
    """
    print("STARTING WITH M = {}".format(config[M]))
    error_rates = []  # List of error rates for each fold
    # In order to get error bars we need to do the prototyping k times
    for fold in range(1, config[K] + 1):
        # Do prototyping
        Xtrain_rand, ytrain_rand = config[SAMPLER](Xtrain, ytrain, config[M])

        # To reduce dimensionality and run time we PCA transform the input
        pca = PCA(n_components=config[N_COMPONENTS])
        Xtrain_rand = pca.fit_transform(Xtrain_rand)
        Xtest_transformed = pca.transform(Xtest)

        # Initialize a model
        model = get_model(Xtrain_rand, ytrain_rand, algorithm=config[ALGORITHM], metric=config[METRIC], n_neighbors=1)

        # Test model
        predictions = model.predict(Xtest_transformed)
        error_rate = sum(predictions != ytest) / ytest.shape[0]
        error_rates.append(error_rate)
        print("FOLD: {}, ERROR RATE: {}".format(fold, error_rate))
    return error_rates


def stats(error_rates_pr_m, path=None):
    file = open(path, "w")
    writer = csv.writer(file, delimiter=' ')
    writer.writerow(["M", "mean", "std", "maxi", "mini", "interval"])
    avgs, stds = [], []# For graphing
    for m, error_rates in error_rates_pr_m.items():
        mean = np.mean(error_rates)
        avgs.append(mean)
        std = np.std(error_rates)
        stds.append(std)
        maxi = np.max(error_rates)
        mini = np.min(error_rates)
        confidence_interval = st.t.interval(0.95, len(error_rates) - 1, loc=mean, scale=st.sem(error_rates))
        writer.writerow([m, mean, std, maxi, mini, confidence_interval])
        print(
            "MEAN: {} \nSTD: {} \nMAX: {} \nMIN: {} \nCONF_INT: {}\n".format(mean, std, maxi, mini,
                                                                             confidence_interval))
    file.close()

    return sorted(list(error_rates_pr_m.keys())), avgs, stds




if __name__ == '__main__':
    config = {
        "ALGORITHM": "kd_tree",  # Algorithm used to make 1-NN model more efficient
        "METRIC": "euclidean",  # Distance metric in the model 1-NN model
        "N_COMPONENTS": 40,  # Number of components to use for the PCA
        "K": 5,  # Number of folds
    }
    # Load data
    Xtrain, ytrain = load_data("./", mode="train")
    Xtest, ytest = load_data("./", mode="t10k")

    error_rates_pr_m_random = dict()
    error_rates_pr_m_kmeans = dict()
    for m in [50, 100, 500, 1000, 5000, 10000]:
        config[M] = m

        config[SAMPLER] = uniform_random_selection
        error_rates_pr_m_random[m] = test(Xtrain, ytrain, Xtest, ytest, config)

        config[SAMPLER] = k_means_sampler
        error_rates_pr_m_kmeans[m] = test(Xtrain, ytrain, Xtest, ytest, config)

    ms, avgs_r, stds_r = stats(error_rates_pr_m_random, path="./random_results.csv")
    ms, avgs_k, stds_k = stats(error_rates_pr_m_kmeans, path="./k_means_results.csv")

    # Graph
    plt.xlabel("M")
    plt.ylabel("Error Rate")
    plt.title("Error Rate using M prototypes")
    plt.errorbar(ms, avgs_r, stds_r, label="Uniform Random Selection")
    plt.errorbar(ms, avgs_k, stds_k, label="K-Means Prototyping")
    plt.legend(loc="best")
    plt.savefig("./graph.png")
    plt.show()
