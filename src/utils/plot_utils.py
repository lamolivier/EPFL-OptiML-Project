import math

import matplotlib.pyplot as plt
import numpy as np


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
def plot_accuracies(test_accuracies_dfw, test_accuracies_bcd, filename):
    te_mean_dfw, te_upper_dfw, te_lower_dfw = build_conf_interval(test_accuracies_dfw)
    te_mean_bcd, te_upper_bcd, te_lower_bcd = build_conf_interval(test_accuracies_bcd)
    x = range(1, len(te_mean_dfw) + 1)

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 15})
    plt.plot(x, te_mean_dfw, linewidth=2, label='DFW')  # mean curve.
    plt.plot(x, te_mean_bcd, linewidth=2, color='g', label='BCD')
    plt.fill_between(x, te_lower_dfw, te_upper_dfw, color='b', alpha=.1)
    plt.fill_between(x, te_lower_bcd, te_upper_bcd, color='g', alpha=.1)

    plt.legend()
    plt.ylim((0.65, 1))
    plt.ylabel('Average accuracy')
    plt.xlabel("Number of epochs")
    plt.title('Accuracy vs number of epochs')

    plt.savefig(filename, orientation='landscape')
    plt.show()


def plot_results(data_dict, ylabel, title, filename):
    labels = ['D=500', 'D=1000', 'D=1500']
    bcd_data = np.mean(data_dict['BCD'], axis=1)
    dfw_data = np.mean(data_dict['DFW'], axis=1)
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 7))
    fig
    rects1 = ax.bar(x - width / 2, bcd_data, width, label='BCD')
    rects2 = ax.bar(x + width / 2, dfw_data, width, label='DFW')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()
    fig.savefig(filename, orientation='landscape')
