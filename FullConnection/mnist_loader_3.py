import struct
import numpy as np


def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1), arg)), args
    ))


def ImageLoader(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(
            fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        images[i] /= 255
        offset += struct.calcsize(fmt_image)
    im = []
    for i in range(num_images):
        im.append(images[i].reshape(784, 1))
    return im


def LabelLoader(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    Labels = []
    l = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for i in range(num_images):
        l = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        l[int(labels[i])] = 0.9
        Labels.append(np.array(l).reshape(10, 1))
    return Labels


def get_train_data():
    image = ImageLoader('MNIST/train-images.idx3-ubyte')
    label = LabelLoader('MNIST/train-labels.idx1-ubyte')
    return image, label


def get_test_data():
    image_loader = ImageLoader('MNIST/t10k-images.idx3-ubyte')
    label_loadr = LabelLoader('MNIST/t10k-labels.idx1-ubyte')
    return image_loader, label_loadr


def load_data():
    train_data, train_labels = transpose(get_train_data())
    test_data, test_labels = transpose(get_test_data())
    return train_data, train_labels, test_data, test_labels
