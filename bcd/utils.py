import torch
import torch.nn as nn
import numpy as np
from .helpers import *
from .data_utils import *
from bcd.ModelBCD import ModelBCD
import time
from .metrics import *


def full_train_test(optimizer, N_train, N_test, n_iter = 5, n_epochs =50, l1_size = 200, l2_size = 200, l3_size = 200, gamma = 1, alpha = 1, rho = 1, verbose = False):
    
    
    # Init metrics arrays
    train_loss_matrix = []
    validation_loss_matrix = []
    
    accuracy_array = []
    time_array = []
    
    mnist_trainset, mnist_testset = generate_pair_sets()
    
    for i in range(1, n_iter + 1):
        print("Iteration %d" % i)
        
        start_time = time.time()
        
        if (optimizer == "BCD"):
            
            train_input, train_target, y_train_1hot, test_input, test_target, y_test_1hot = preprocess_data(mnist_trainset, mnist_testset, N_train, N_test)
            model = ModelBCD(train_input.size()[0], l1_size, l2_size, l3_size, 10, gamma, alpha, rho)
            tr_losses, te_losses, tr_acc, te_acc = model.train(n_epochs, train_input,train_target,y_train_1hot, test_input, test_target, y_test_1hot, verbose)
        
        else: 
            
        
        train_loss_matrix.append(tr_losses)
        validation_loss_matrix.append(te_losses)
        
        end_time = time.time()
        
        # Compute test accuracy
        acc = accuracy(model, N_train, mnist_trainset, mnist_testset)
        
        accuracy_array.append(acc)
        time_array.append(end_time - start_time)
        
    plot_losses(train_loss_matrix, validation_loss_matrix)
    plot_accuracy(accuracy_array)
    
    acc_mean, acc_std = extract_mean_std(accuracy_array)
    time_mean, time_std = extract_mean_std(time_array)

    print("Accuracy: %.3f +/- %.3f" % (acc_mean, acc_std))
    print("Iteration time:  %.3f +/- %.3f seconds" % (time_mean, time_std))
