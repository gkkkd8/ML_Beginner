from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np


def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test = np.random.normal(x_test)
    x_train, x_test = x_train / 255, x_test / 255
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = [np.reshape(x, (784, 1)) for x in x_train]
    x_test = [np.reshape(x, (784, 1)) for x in x_test]
    y_train = [np.reshape(x, (10, 1)) for x in y_train]
    y_test = [np.reshape(x, (10, 1)) for x in y_test]

    return x_train, y_train, x_test, y_test


# load_data()
