import numpy as np
import mnist_loader_2 as data_loader
from datetime import datetime


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return np.multiply(output, (1 - output))


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        # delta_array: 从上一层传递过来的误差项
        self.delta = self.activator.backward(
            self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1], SigmoidActivator())
            )

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


def evaluate(network, test_data, test_labels):
    error = 0
    total = len(test_data)
    for i in range(total):
        label = np.argmax(test_labels[i])
        t = network.predict(test_data[i])
        predict = np.argmax(t)
        if label != predict:
            error += 1
    return float(error)/float(total)


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data, train_labels, test_data, test_labels = data_loader.load_data()
    print(datetime.now())
    network = Network([784, 30, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data, 0.3, 1)
        print('%s epoch %d finished' % (datetime.now(), epoch))
        if epoch % 1 == 0:
            error_ratio = evaluate(network, test_data, test_labels)
            print('%s after epoch %d, error ratio is %f' %
                  (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == "__main__":
    train_and_evaluate()
