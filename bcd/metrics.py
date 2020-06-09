import torch
import math
from .data_utils import *
import numpy as np

def extract_mean_std(array):
    mean = sum(array) / len(array)
    std = math.sqrt(sum([pow(x - mean, 2) for x in array]) / len(array))

    return mean, std


# Compute accuracy for digit recognition.
def accuracy(model, N, mnist_trainset, mnist_testset):
    _, _, _, x_test, y_test, _ = preprocess_data(mnist_trainset, mnist_testset, 0, N)
    
    test_output = model.forward(x_test)
    pred_test = torch.argmax(test_output, dim=0)
    
    correct_test = pred_test == y_test
    acc = np.mean(correct_test.numpy())
    
    return acc
