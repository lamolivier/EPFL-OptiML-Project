import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the train data and test data from MNIST dataset
def get_sets():

    ts = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0,), (1,))])

    data_dir = '../data'
    train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True, transform=ts)
    test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True, transform=ts)

    return train_set, test_set

# Generate dataloader from randomly sampled data of size N for training and N_test for testing
def generate_pair_sets(train_set, test_set, N, N_test, batch_size=1):

    # N=0 when we only want to generate a test loader
    if N != 0:

        # Sample random indexes without repetition
        train_idx = random.sample(range(60000), k=N)

        # Create a subset containing those samples
        trainset = Subset(train_set, train_idx)

        # Instantiate a DataLoader from this subset
        train_data = DataLoader(trainset, batch_size=batch_size)

    else:
        train_data = None

    test_idx = random.sample(range(10000), k=N_test)
    testset = Subset(test_set, test_idx)
    test_data = DataLoader(testset, batch_size=batch_size)

    return train_data, test_data

# Preprocess the data for BCD algorithm
def preprocess_data(train_data, test_data, N, N_test):

    # Manipulate train data
    nb_ch, nb_row, nb_col = 1, 28, 28
    K = 10
    if (train_data == None):
        x_train = y_train = y_train_one_hot = None

    else:
        x_train = torch.empty((N, nb_ch * nb_row * nb_col), device=device)
        y_train = torch.empty(N, dtype=torch.long)

        # Iterate through the train loader
        for i, j in enumerate(train_data):

            # Flatten [1,28, 28] image into one row
            x_train[i, :] = j[0].reshape(1, nb_ch * nb_row * nb_col)

            # Retrieve class labels
            y_train[i] = j[1]

        x_train = x_train.t()

        # Convert into one-hot labels
        y_train_one_hot = torch.zeros(N, K).scatter_(1, y_train.reshape(N, 1), 1)
        y_train_one_hot = y_train_one_hot.t().to(device=device)
        y_train = y_train.to(device=device)

    # Manipulate test data in the same way as train data
    x_test = torch.empty((N_test, nb_ch * nb_row * nb_col), device=device)
    y_test = torch.empty(N_test, dtype=torch.long)

    for i, j in enumerate(test_data):
        x_test[i, :] = j[0].reshape(1, nb_ch * nb_row * nb_col)
        y_test[i] = j[1]

    x_test = x_test.t()
    y_test_one_hot = torch.zeros(N_test, K).scatter_(1, y_test.reshape(N_test, 1), 1)
    y_test_one_hot = y_test_one_hot.t().to(device=device)
    y_test = y_test.to(device=device)

    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot
