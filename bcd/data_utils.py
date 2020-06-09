import torch
from torchvision import datasets, transforms
import random

def generate_pair_sets():
    
    ts = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0,), (1,))])
    data_dir = '../data'
    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True, transform = ts)
    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True, transform = ts)

    return train_set, test_set
           
    
def preprocess_data(trainset, testset, N, N_test):
    # Manipulate train set
    nb_ch, nb_row, nb_col = trainset[0][0].size()
    K = 10
    if (N == 0):
        x_train = y_train = y_train_one_hot = None
        
    else:
        x_train = torch.empty((N, nb_ch * nb_row * nb_col))
        y_train = torch.empty(N, dtype=torch.long)
        for j in range(N):
            #Generate random train set of size N
            i = random.randrange(60000)
            #Flatten image 
            x_train[j, :] = trainset[i][0].reshape(1, nb_ch * nb_row * nb_col)
            #Retrieve labels
            y_train[j] = trainset[i][1]
        
        x_train = x_train.t()
    
        #Convert in one-hot labels
        y_train_one_hot = torch.zeros(N, K).scatter_(1, y_train.reshape(N, 1), 1)
        y_train_one_hot = y_train_one_hot.t()

    # Manipulate test set
    x_test = torch.empty((N_test, nb_ch * nb_row * nb_col))
    y_test = torch.empty(N_test, dtype=torch.long)
    for j in range(N_test):
        i = random.randrange(10000)
        x_test[j, :] = testset[i][0].reshape(1,  nb_ch * nb_row * nb_col)
        y_test[j] = testset[i][1]
        
    x_test = x_test.t()
    y_test_one_hot = torch.zeros(N_test, K).scatter_(1, y_test.reshape(N_test, 1), 1)
    y_test_one_hot = y_test_one_hot.t()

    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot