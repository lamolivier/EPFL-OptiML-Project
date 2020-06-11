import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import DataLoader, Subset

def get_sets():
    ts = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0,), (1,))])
    
    data_dir = '../data'
    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True, transform = ts)
    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True, transform = ts)
    
    return train_set, test_set
    

def generate_pair_sets(train_set, test_set, N, N_test, batch_size = 1):
    
    if N != 0:
        
        train_idx = [random.randrange(60000) for i in range(N)]
        trainset = Subset(train_set, train_idx)
        train_data = DataLoader(trainset,batch_size=batch_size)
        
    else: 
        train_data = None
        
    test_idx = [random.randrange(10000) for i in range(N_test)]
    testset = Subset(test_set, test_idx)
    test_data = DataLoader(testset,batch_size=batch_size)

    return train_data, test_data
           
    
def preprocess_data(train_data, test_data, N, N_test):
    # Manipulate train set
    nb_ch, nb_row, nb_col = 1, 28, 28
    K = 10
    if (train_data == None):
        x_train = y_train = y_train_one_hot = None
        
    else:
        x_train = torch.empty((N, nb_ch * nb_row * nb_col))
        y_train = torch.empty(N, dtype=torch.long)
        for i,j in enumerate(train_data):
            #Flatten image 
            x_train[i, :] = j[0].reshape(1, nb_ch * nb_row * nb_col)
            #Retrieve labels
            y_train[i] = j[1]
        
        x_train = x_train.t()
    
        #Convert in one-hot labels
        y_train_one_hot = torch.zeros(N, K).scatter_(1, y_train.reshape(N, 1), 1)
        y_train_one_hot = y_train_one_hot.t()

    # Manipulate test set
    x_test = torch.empty((N_test, nb_ch * nb_row * nb_col))
    y_test = torch.empty(N_test, dtype=torch.long)
    for i,j in enumerate(test_data):
        x_test[i, :] = j[0].reshape(1,  nb_ch * nb_row * nb_col)
        y_test[i] = j[1]
        
    x_test = x_test.t()
    y_test_one_hot = torch.zeros(N_test, K).scatter_(1, y_test.reshape(N_test, 1), 1)
    y_test_one_hot = y_test_one_hot.t()

    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot