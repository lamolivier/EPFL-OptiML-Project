import torch
import torch.nn as nn
import numpy as np
from .helpers import *
from .data_utils import *
from bcd.ModelBCD import ModelBCD
import time
from .metrics import *
from dfw.ModelDFW import ModelDFW
from torch.utils.data import DataLoader


def full_train_test(optimizer, N_train, N_test, n_iter, n_epochs, batch_size = 0, d1 = 200, d2 = 200, d3 = 200, gamma = 1, alpha = 1, rho = 1, verbose = False):
    
    # Init metrics arrays
    train_loss_matrix = []
    validation_loss_matrix = []
    
    accuracy_test_array = []
    accuracy_train_array = []
    time_array = []
    
    mnist_trainset, mnist_testset = generate_pair_sets()
    
    d0 =  1*28*28
    
    for i in range(1, n_iter + 1):
        print("Iteration %d" % i)

        start_time = time.time()
        
        if (optimizer == "BCD"):
            train_input, train_target, y_train_1hot, test_input, test_target, y_test_1hot = preprocess_data(mnist_trainset, mnist_testset, N_train, N_test)
            model = ModelBCD(d0, d1, d2, d3, 10, gamma, alpha, rho)
            tr_losses, te_losses, tr_acc, te_acc = model.train(n_epochs, train_input,train_target,y_train_1hot, test_input, test_target, y_test_1hot, verbose = verbose)
        else:
            train_data = DataLoader(mnist_trainset,batch_size=batch_size)
            test_data = DataLoader(mnist_testset,batch_size=batch_size)
            model = ModelDFW(d0, d1, d2, d3, 10)
            tr_losses, te_losses, tr_acc, te_acc = model.train(train_data, test_data, n_epochs, verbose = verbose)

        train_loss_matrix.append(tr_losses)
        validation_loss_matrix.append(te_losses)
        
        end_time = time.time()
        
        if (optimizer == "BCD"):
        # Compute test accuracy
            acc_test = accuracy(model, N_test, mnist_trainset, mnist_testset)
            acc_train = accuracy(model, N_train, mnist_trainset, mnist_testset)
        else: 
            test_data = DataLoader(mnist_testset,batch_size=batch_size, shuffle = True)
            acc_test = model.test(test_data, batch_size)
            acc_train = model.test(train_data, batch_size)
        
        accuracy_test_array.append(acc_test)
        accuracy_train_array.append(acc_train)
        
        time_array.append(end_time - start_time)
    
    plot_losses(train_loss_matrix, validation_loss_matrix)
    plot_accuracy(accuracy_test_array, accuracy_train_array)
    
    acc_mean, acc_std = extract_mean_std(accuracy_test_array)
    time_mean, time_std = extract_mean_std(time_array)

    print("Accuracy: %.3f +/- %.3f" % (acc_mean, acc_std))
    print("Iteration time:  %.3f +/- %.3f seconds" % (time_mean, time_std))
