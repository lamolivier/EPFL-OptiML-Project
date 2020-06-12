import math

import numpy as np

from .data_utils import *


def extract_mean_std(array):
    mean = sum(array) / len(array)
    std = math.sqrt(sum([pow(x - mean, 2) for x in array]) / len(array))

    return mean, std


# Compute accuracy for digit recognition.
def accuracy(model, N, train_set, test_set):
    train_data, test_data = generate_pair_sets(train_set, test_set, 0, N)
    _, _, _, x_test, y_test, _ = preprocess_data(train_data, test_data, 0, N)

    test_output = model.forward(x_test)
    pred_test = torch.argmax(test_output, dim=0)

    correct_test = pred_test == y_test
    acc = np.mean(correct_test.numpy())

    return acc
