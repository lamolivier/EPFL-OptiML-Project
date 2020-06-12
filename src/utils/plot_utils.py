import matplotlib.pyplot as plt
import numpy as np
import math

# Computes the mean and standard deviation of values in an array
def extract_mean_std(array):
    mean = sum(array) / len(array)
    std = math.sqrt(sum([pow(x - mean, 2) for x in array]) / len(array))

    return mean, std


# Build confidence interval from multiple trainings on different data
def build_conf_interval(all_losses):
    mean = np.mean(all_losses, axis=0)
    std = np.std(all_losses, axis=0)
    lower = mean - 2 * std
    upper = mean + 2 * std
    return mean, upper, lower


# Build train and test losses during training with confidence interval
def plot_accuracies(all_train_accuracies, all_test_accuracies):
    tr_mean, tr_upper, tr_lower = build_conf_interval(all_train_accuracies)
    te_mean, te_upper, te_lower = build_conf_interval(all_test_accuracies)
    x = range(1, len(tr_mean) + 1)

    plt.figure(figsize=(15, 8))
    plt.plot(x, tr_mean, linewidth=2, label='Train accuracy')  # mean curve.
    plt.plot(x, te_mean, linewidth=2, color='g', label='Test accuracy')
    plt.fill_between(x, tr_lower, tr_upper, color='b', alpha=.1)
    plt.fill_between(x, te_lower, te_upper, color='g', alpha=.1)

    plt.legend()
    plt.ylabel('Average accuracy')
    plt.xlabel("Number of epochs")
    plt.title('Accuracy vs number of epochs')
    plt.show()


# Build a boxplot of the accuracy for multiple iterations.
def plot_accuracy(test_accuracies):
    print('Test accuracy mean = ' + str(np.mean(test_accuracies)))
    plt.figure(figsize=(15, 8))
    plt.boxplot(test_accuracies)
    plt.xticks([1], ['Test accuracy distribution'])
    plt.title('Test Accuracy distribution')
    plt.show()
