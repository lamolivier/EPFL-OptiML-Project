import torch
from bcd.helpers import *
from bcd.ModelBCD import ModelBCD
import time
from bcd.metrics import *
from dfw.ModelDFW import ModelDFW
from torch.utils.data import DataLoader


def full_train_test(optimizer, N_train, N_test, n_iter, n_epochs, batch_size=0, d1=200, d2=200, d3=200, gamma=1,
                    alpha=1, rho=1, verbose=False):
    # Init metrics arrays

    accuracy_test_array = []
    accuracy_train_array = []
    time_array = []

    mnist_trainset, mnist_testset = generate_pair_sets()

    d0 = 1 * 28 * 28

    for i in range(1, n_iter + 1):
        print("Iteration %d" % i)

        if (optimizer == "BCD"):
            train_input, train_target, y_train_1hot, test_input, test_target, y_test_1hot = preprocess_data(
                mnist_trainset, mnist_testset, N_train, N_test)
            start_time = time.time()
            model = ModelBCD(d0, d1, d2, d3, 10, gamma, alpha, rho)
            tr_acc, te_acc = model.train(n_epochs, train_input, train_target, y_train_1hot,
                                                               test_input, test_target, y_test_1hot, verbose=verbose)
        else:
            train_data = DataLoader(mnist_trainset, batch_size=batch_size)
            test_data = DataLoader(mnist_testset, batch_size=batch_size)
            start_time = time.time()
            model = ModelDFW(d0, d1, d2, d3, 10)
            tr_acc, te_acc = model.train(train_data, test_data, n_epochs, verbose=verbose)

        end_time = time.time()

        if (optimizer == "BCD"):
            # Compute test accuracy
            acc_test = accuracy(model, N_test, mnist_trainset, mnist_testset)
            acc_train = accuracy(model, N_train, mnist_trainset, mnist_testset)
        else:
            test_data = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
            acc_test = model.test(test_data, batch_size)
            acc_train = model.test(train_data, batch_size)

        accuracy_test_array.append(acc_test)
        accuracy_train_array.append(acc_train)

        time_array.append(end_time - start_time)

    plot_accuracies(accuracy_train_array, accuracy_test_array)

    acc_mean, acc_std = extract_mean_std(accuracy_test_array)
    time_mean, time_std = extract_mean_std(time_array)

    print("Accuracy: %.3f +/- %.3f" % (acc_mean, acc_std))
    print("Iteration time:  %.3f +/- %.3f seconds" % (time_mean, time_std))
