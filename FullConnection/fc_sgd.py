import random
import numpy as np
import mnist_loader as data_loader


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class FCN(object):
    def __init__(self, sizes):
        self._num_layers = len(sizes)  # 记录神经网络的层数
        self._biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self._weights = [np.random.randn(y, x)
                         for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self._biases, self._weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 在每一次迭代之前，都将训练数据集进行随机打乱，然后每次随机选取若干个小批量训练数据集
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            # 每次训练迭代周期中要使用完全部的小批量训练数据集
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # 如果test_data被指定，那么在每一轮迭代完成之后，都对测试数据集进行评估，计算有多少样本被正确识别了
            if test_data:
                print("Epoch %d: accuracy rate: %.2f%%" %
                      (j, self.evaluate(test_data)/n_test*100))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        for x, y in mini_batch:
            # 反向传播算法，运用链式法则求得对b和w的偏导
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 对小批量训练数据集中的每一个求得的偏导数进行累加
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降得出的规则来更新权重和偏置向量
        self._weights = [w - (eta / len(mini_batch)) * nw
                         for w, nw in zip(self._weights, nabla_w)]
        self._biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self._biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        activation = x
        activations = [x]  # 记录每一层输入
        zs = []  # 记录每一层输出
        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1]-y) * sigmoid_prime(zs[-1])  # 输出层delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self._weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


if __name__ == "__main__":
    train_input, train_result, test_input, test_result = data_loader.load_data()
    training_data = list(zip(train_input, train_result))
    test_data = list(zip(test_input, test_result))
    fc = FCN([784, 30, 10])
    fc.SGD(training_data, 20, 10, 3.0, test_data=test_data)


# 单层采用30，两层采用80,10
