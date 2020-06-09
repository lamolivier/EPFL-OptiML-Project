import numpy as np
import torch
import matplotlib.pyplot as plt


# Build confidence interval from multiple trainings on different data
def build_conf_interval(all_losses):
    mean = np.mean(all_losses, axis=0)
    std = np.std(all_losses, axis=0)
    lower = mean - 2 * std
    upper = mean + 2 * std
    return mean, upper, lower


# Build train and test losses during training with confidence interval
def plot_losses(all_train_losses, all_test_losses):
    tr_mean, tr_upper, tr_lower = build_conf_interval(all_train_losses)
    te_mean, te_upper, te_lower = build_conf_interval(all_test_losses)
    x = range(1, len(tr_mean) + 1)

    plt.figure(figsize=(15, 8))
    plt.plot(x, tr_mean, linewidth=2, label='Train loss')  # mean curve.
    plt.plot(x, te_mean, linewidth=2, color='g', label='Test loss')
    plt.fill_between(x, tr_lower, tr_upper, color='b', alpha=.1)
    plt.fill_between(x, te_lower, te_upper, color='g', alpha=.1)

    plt.legend()
    plt.ylabel('Average MSELoss')
    plt.xlabel("Number of epochs")
    plt.title('MSELoss vs number of epochs')
    plt.show()

# Build a boxplot of the accuracy for multiple iterations.
def plot_accuracy(all_accuracies):
    print('Test accuracy mean = ' + str(np.mean(all_accuracies)))
    plt.figure(figsize=(15, 8))
    plt.boxplot(all_accuracies)
    plt.title('Test Accuracy distribution')
    plt.show()