import os
import gzip
import numpy as np
import matplotlib.pyplot as plt


def load_data(path, mode='train'):
    """
    Load  MNIST data.
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


if __name__ == '__main__':
    Xtrain, ytrain = load_data("./", mode="train")
    Xtest, ytest = load_data("./", mode="t10k")

    Xtrain_rand, ytrain_rand = uniform_random_selection(Xtrain, ytrain, 100)
