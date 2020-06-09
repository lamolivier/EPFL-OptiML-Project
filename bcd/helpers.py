import numpy as np
import torch
import matplotlib.pyplot as plt


def preprocess_data(trainset, testset):
    # Manipulate train set
    nb_ch, nb_row, nb_col = trainset[0][0].size()
    N = len(trainset)
    K = 10
    x_train = torch.empty((N, nb_ch * nb_row * nb_col))
    y_train = torch.empty(N, dtype=torch.long)
    for i in range(N):
        #Flatten image 
        x_train[i, :] = trainset[i][0].reshape(1, nb_ch * nb_row * nb_col)
        #Retrieve labels
        y_train[i] = trainset[i][1]
        
    x_train = x_train.t()
    
    #Convert in one-hot labels
    y_train_one_hot = torch.zeros(N, K).scatter_(1, y_train.reshape(N, 1), 1)
    y_train_one_hot = y_train_one_hot.t()

    # Manipulate test set
    N_test = len(testset)
    x_test = torch.empty((N_test, nb_ch * nb_row * nb_col))
    y_test = torch.empty(N_test, dtype=torch.long)
    for i in range(N_test):
        x_test[i, :] = testset[i][0].reshape(1,  nb_ch * nb_row * nb_col)
        y_test[i] = testset[i][1]
        
    x_test = x_test.t()
    y_test_one_hot = torch.zeros(N_test, K).scatter_(1, y_test.reshape(N_test, 1), 1)
    y_test_one_hot = y_test_one_hot.t()

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

    
def build_conf_interval(all_losses):
    mean = np.mean(all_losses, axis=0)
    std = np.std(all_losses, axis=0)
    lower = mean - 2 * std
    upper = mean + 2 * std
    return mean, upper, lower


# Build train and test losses during training with confidence interval
def plot_losses(all_train_losses, all_test_losses, n_iter):
    
    x = range(1, n_iter + 1)
        
    plt.figure(figsize=(15, 8))
    plt.plot(x, all_train_losses, linewidth=2, label='Train loss')  # mean curve.
    plt.plot(x, all_test_losses, linewidth=2, color='g', label='Test loss')

    plt.legend()
    plt.ylabel('Average MSE')
    plt.xlabel("Number of epochs")
    plt.title('MSE vs number of epochs')
    plt.show()