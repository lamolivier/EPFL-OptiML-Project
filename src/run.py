import time
from bcd.ModelBCD import ModelBCD
from utils.plot_utils import *
from dfw.ModelDFW import ModelDFW
from utils.data_utils import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.append('my/path/to/module/folder')

def full_train_test(optimizer, N_train, N_test, n_iter, n_epochs, batch_size=1, d1=200, d2=200, d3=200, gamma=1,
                    alpha=1, rho=1, verbose=False):
    
    if (optimizer == "BCD"):
        batch_size = 1
    
    # Initialize metrics arrays
    accuracy_test_array = []
    accuracy_train_array = []

    tr_acc_evolution = []
    te_acc_evolution = []

    time_array = []
    
    # Generate the sets
    train_set, test_set = get_sets()
    
    # Generate train and test DataLoader
    train_data, test_data = generate_pair_sets(train_set, test_set, N_train, N_test, batch_size)

    d0 = 1 * 28 * 28

    for i in range(1, n_iter + 1):
        print("Iteration %d" % i)
        
        if (optimizer == "BCD"):
            train_input, train_target, y_train_1hot, test_input, test_target, y_test_1hot = preprocess_data(train_data,
                                                                                                            test_data,
                                                                                                            N_train,
                                                                                                            N_test)
            # Instantiate the model
            model = ModelBCD(d0, d1, d2, d3, 10, gamma, alpha, rho)
            
            start_time = time.time()
            
            # Train the model
            tr_acc, te_acc = model.train(n_epochs, train_input, train_target, y_train_1hot,
                                         test_input, test_target, y_test_1hot, verbose=verbose)
        else:
            # Instantiate the model
            model = ModelDFW(d0, d1, d2, d3, 10)
            
            start_time = time.time()
            
            # Train the model
            tr_acc, te_acc = model.train(train_data, test_data, n_epochs, verbose=verbose)

        end_time = time.time()
        
        # Store the train and test accuracy for each epoch of the iteration
        tr_acc_evolution.append(tr_acc)
        te_acc_evolution.append(te_acc)
        
        # Computes test accuracy with "new" data
        if (optimizer == "BCD"):
            acc_test = model.test(N_test, train_set, test_set)
        else:
            _, test_data = generate_pair_sets(train_set, test_set, N_train, N_test, batch_size)
            acc_test = model.test(test_data, batch_size)
            
        # Store the test accuracies for each iteration
        accuracy_test_array.append(acc_test)
        
        # Store the train accuracies for each iteration
        accuracy_train_array.append(tr_acc[n_epochs - 1])
        
        # Store the duration of the iteraton
        time_array.append(end_time - start_time)
    
    # Plot the training and validation accuracy
    plot_accuracies(tr_acc_evolution, te_acc_evolution)
    
    # Plot the test accuracy
    plot_accuracy(accuracy_test_array)
    
    # Compute the mean and std of the test accuracies
    acc_mean, acc_std = extract_mean_std(accuracy_test_array)
    
    # Compute the mean and std of the duration an iteraion
    time_mean, time_std = extract_mean_std(time_array)

    print("Accuracy: %.3f +/- %.3f" % (acc_mean, acc_std))
    print("Iteration time:  %.3f +/- %.3f seconds" % (time_mean, time_std))

    
if __name__ == "__main__":
    
    full_train_test("BCD",30000, 5000, 1, 3 ,d1 = 200, d2 = 200, d3 = 200, verbose = True)