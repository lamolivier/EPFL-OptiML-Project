import numpy as np
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import generate_pair_sets, preprocess_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Forward step
def feed_forward(Wn, bn, Vn_1, n_samples):
    Un = torch.addmm(bn.repeat(1, n_samples), Wn, Vn_1)
    Vn = nn.ReLU()(Un)
    return Un, Vn

# Update step of the state variable V
def updateVn(Un, Un1, Wn1, bn1, rho, gamma):
    d = Wn1.size()[1]
    I = torch.eye(d, device=device)
    Vn = nn.ReLU()(Un)
    col_Un1 = Un1.size()[1]
    Vs = torch.mm(torch.inverse(rho * (torch.mm(torch.t(Wn1), Wn1)) + gamma * I),
                  rho * torch.mm(torch.t(Wn1), Un1 - bn1.repeat(1, col_Un1)) + gamma * Vn)
    return Vs

# Update step of the weights and bias
def updateWn(Un, Vn_1, Wn, bn, alpha, rho):
    d, N = Vn_1.size()
    I = torch.eye(d, device=device)
    col_Un = Un.size()[1]
    Ws = torch.mm(alpha * Wn + rho * torch.mm(Un - bn.repeat(1, col_Un), torch.t(Vn_1)),
                  torch.inverse(alpha * I + rho * (torch.mm(Vn_1, torch.t(Vn_1)))))
    bs = (alpha * bn + rho * torch.sum(Un - torch.mm(Wn, Vn_1), dim=1).reshape(bn.size())) / (rho * N + alpha)
    return Ws, bs

# Approximative ReLU function
def relu_prox(a, b, gamma, d, N):
    x = (a + gamma * b) / (1 + gamma)
    y = torch.min(b, torch.zeros(d, N, device=device))
    val = torch.where(a + gamma * b < 0, y, torch.zeros(d, N, device=device))
    val = torch.where(
        ((a + gamma * b >= 0) & (b >= 0)) | ((a * (gamma - np.sqrt(gamma * (gamma + 1))) <= gamma * b) & (b < 0)), x,
        val)
    val = torch.where((-a <= gamma * b) & (gamma * b <= a * (gamma - np.sqrt(gamma * (gamma + 1)))), b, val)
    return val

# Update of one block
def block_update(Wn, bn, Wn_1, bn_1, Un, Vn_1, Un_1, Vn_2, dn_1, alpha, gamma, rho, dim):
    # update W(n) and b(n)
    Wn, bn = updateWn(Un, Vn_1, Wn, bn, alpha, rho)
    # update V(n-1)
    Vn_1 = updateVn(Un_1, Un, Wn, bn, rho, gamma)
    # update U(n-1)
    Un_1 = relu_prox(Vn_1, (rho * torch.addmm(bn_1.repeat(1, dim), Wn_1, Vn_2) +
                            alpha * Un_1) / (rho + alpha), (rho + alpha) / gamma, dn_1, dim)
    return Wn, bn, Vn_1, Un_1


class ModelBCD:

    def __init__(self, d0, d1, d2, d3, classes, gamma, alpha, rho):
        # Layer 1
        self.fc1 = nn.Linear(d0, d1).to(device=device)
        self.b1 = self.fc1.bias.reshape(d1, 1).data
        self.w1 = self.fc1.weight.data
        # Layer 2
        self.fc2 = nn.Linear(d1, d2).to(device=device)
        self.b2 = self.fc2.bias.reshape(d2, 1).data
        self.w2 = self.fc2.weight.data
        # Layer 3
        self.fc3 = nn.Linear(d2, d3).to(device=device)
        self.b3 = self.fc3.bias.reshape(d3, 1).data
        self.w3 = self.fc3.weight.data
        # Layer 4
        self.fc4 = nn.Linear(d3, classes).to(device=device)
        self.b4 = self.fc4.bias.reshape(classes, 1).data
        self.w4 = self.fc4.weight.data

        self.gamma = gamma
        self.alpha = alpha
        self.rho = rho
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.classes = classes

    # Initilization of the auxiliary varibles U and V
    def init_aux_params(self, x):
        n_samples = x.size()[1]
        self.V0 = x
        self.U1, self.V1 = feed_forward(self.w1, self.b1, self.V0, n_samples)
        self.U2, self.V2 = feed_forward(self.w2, self.b2, self.V1, n_samples)
        self.U3, self.V3 = feed_forward(self.w3, self.b3, self.V2, n_samples)
        self.U4 = torch.addmm(self.b4.repeat(1, n_samples), self.w4, self.V3)
        self.V4 = self.U4  # as sigma_4 = Id
    
    # Forward propagation
    def forward(self, x):
        n_samples = x.size()[1]
        V1 = feed_forward(self.w1, self.b1, x, n_samples)[1]
        V2 = feed_forward(self.w2, self.b2, V1, n_samples)[1]
        V3 = feed_forward(self.w3, self.b3, V2, n_samples)[1]
        output = torch.addmm(self.b4.repeat(1, n_samples), self.w4, V3)
        return output
    
    # Update all the variables cyclically while fixing the remaining blocks
    def update_params(self, y_one_hot, x):
        n_samples = x.size()[1]

        # update V4
        self.V4 = (y_one_hot + self.gamma * self.U4 + self.alpha * self.V4) / (1 + self.gamma + self.alpha)

        # update U4
        self.U4 = (self.gamma * self.V4 + self.rho * (
                torch.mm(self.w4, self.V3) + self.b4.repeat(1, n_samples))) / (
                          self.gamma + self.rho)

        # update W4, b4, V3 and U3
        self.w4, self.b4, self.V3, self.U3 = block_update(self.w4, self.b4,
                                                          self.w3, self.b3, self.U4,
                                                          self.V3, self.U3, self.V2, self.d3,
                                                          self.alpha, self.gamma, self.rho, n_samples)

        # update W3, b3, V2 and U2
        self.w3, self.b3, self.V2, self.U2 = block_update(self.w3, self.b3,
                                                          self.w2, self.b2, self.U3,
                                                          self.V2, self.U2, self.V1, self.d2,
                                                          self.alpha,
                                                          self.gamma, self.rho, n_samples)

        # update W2, b2, V1 and U1
        self.w2, self.b2, self.V1, self.U1 = block_update(self.w2, self.b2,
                                                          self.w1, self.b1, self.U2,
                                                          self.V1, self.U1, x, self.d1, self.alpha,
                                                          self.gamma, self.rho, n_samples)

        # update W1 and b1
        self.w1, self.b1 = updateWn(self.U1, x, self.w1, self.b1, self.alpha, self.rho)

    def train(self, n_epochs, train_input, train_target, y_train_1hot, test_input, test_target, y_test_1hot,
              verbose=False):
        
        # Instantiate loss function 
        criterion = nn.MSELoss()

        # Initialization of the metrics arrays
        tr_losses = []
        te_losses = []
        tr_acc = []
        te_acc = []
        
        # Initialize auxiliary variables
        self.init_aux_params(train_input)

        for e in range(n_epochs):

            self.update_params(y_train_1hot, train_input)

            # Train forward pass
            train_output = self.forward(train_input)
            pred_train = torch.argmax(train_output, dim=0)

            # Test forward pass
            test_output = self.forward(test_input)
            pred_test = torch.argmax(test_output, dim=0)

            # Compute train predictions
            correct_train = pred_train == train_target
            acc_train = np.mean(correct_train.cpu().numpy())
            tr_acc.append(acc_train)

            # Compute test predictions
            correct_test = pred_test == test_target
            acc_test = np.mean(correct_test.cpu().numpy())
            te_acc.append(acc_test)

            # Compute losses
            tr_losses.append(criterion(self.V4, y_train_1hot).cpu().numpy())
            te_losses.append(criterion(test_output, y_test_1hot).cpu().numpy())

            # Print results
            if verbose:
                print(
                    f"Epoch: {e + 1} / {n_epochs} \n Train loss: {tr_losses[e]:.4f} - Test loss:{te_losses[e]:.4f} \n Train acc: {acc_train:.4f} - Test acc: {acc_test:.4f}")

        return tr_acc, te_acc

    
    
    # Compute accuracy for digit recognition using BCD
    def test(self, N, train_set, test_set):

        # Generate "new" data 
        train_data, test_data = generate_pair_sets(train_set, test_set, 0, N)
        _, _, _, x_test, y_test, _ = preprocess_data(train_data, test_data, 0, N)

        # Compute the model's prediction
        test_output = self.forward(x_test)
        pred_test = torch.argmax(test_output, dim=0)

        # Compare the prediction with the real values
        correct_test = pred_test == y_test
        acc = np.mean(correct_test.numpy())

        return acc