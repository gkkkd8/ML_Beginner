import numpy as np
import pickle as cPickle
import gzip


def load_file():
    f = gzip.open('MNIST\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(
        f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)


def load_data():
    tr_d, va_d, te_d = load_file()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    return training_inputs, training_results, test_inputs, test_results


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
