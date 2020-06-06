import numpy as np
import torch
import matplotlib.pyplot as plt


def preprocess_data(trainset, testset):
    # Manipulate train set
    x_d0 = trainset[0][0].size()[0]
    x_d1 = trainset[0][0].size()[1]
    x_d2 = trainset[0][0].size()[2]
    N = len(trainset)
    K = 10
    x_train = torch.empty((N, x_d0 * x_d1 * x_d2))
    y_train = torch.empty(N, dtype=torch.long)
    for i in range(N):
        x_train[i, :] = torch.reshape(trainset[i][0], (1, x_d0 * x_d1 * x_d2))
        y_train[i] = trainset[i][1]
    x_train = torch.t(x_train)
    y_train_one_hot = torch.zeros(N, K).scatter_(1, torch.reshape(y_train, (N, 1)), 1)
    y_train_one_hot = torch.t(y_train_one_hot)

    # Manipulate test set
    N_test = len(testset)
    x_test = torch.empty((N_test, x_d0 * x_d1 * x_d2))
    y_test = torch.empty(N_test, dtype=torch.long)
    for i in range(N_test):
        x_test[i, :] = torch.reshape(testset[i][0], (1, x_d0 * x_d1 * x_d2))
        y_test[i] = testset[i][1]
    x_test = torch.t(x_test)
    y_test_one_hot = torch.zeros(N_test, K).scatter_(1, torch.reshape(y_test, (N_test, 1)), 1)
    y_test_one_hot = torch.t(y_test_one_hot)

    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot


def plot_results(n_iter, acc_train, acc_test, loss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, n_iter + 1), loss, label='Training loss')
    ax.set_title('Three-layer MLP (BCD)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    fig.savefig('results/losses.png', format='png', orientation='landscape')

    # Plot of Accuracies
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, n_iter + 1), acc_train, label='Training accuracy', linewidth=1.5)
    ax.plot(np.arange(1, n_iter + 1), acc_test, label='Validation accuracy', linewidth=1.5)
    ax.set_title('Three-layer MLP (BCD)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')

    fig.savefig('results/accuracies.png', format='png', orientation='landscape')
    print('figures generated and saved in root directory')
