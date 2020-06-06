import numpy as np
import torch
import torch.nn as nn


def feed_forward(weight, bias, activation, dim):
    U = torch.addmm(bias.repeat(1, dim), weight, activation)
    V = nn.ReLU()(U)
    return U, V


def updateV(U1, U2, W, b, rho, gamma):
    _, d = W.size()
    I = torch.eye(d)
    U1 = nn.ReLU()(U1)
    _, col_U2 = U2.size()
    Vstar = torch.mm(torch.inverse(rho * (torch.mm(torch.t(W), W)) + gamma * I),
                     rho * torch.mm(torch.t(W), U2 - b.repeat(1, col_U2)) + gamma * U1)
    return Vstar


def updateWb(U, V, W, b, alpha, rho):
    d, N = V.size()
    I = torch.eye(d)
    _, col_U = U.size()
    Wstar = torch.mm(alpha * W + rho * torch.mm(U - b.repeat(1, col_U), torch.t(V)),
                     torch.inverse(alpha * I + rho * (torch.mm(V, torch.t(V)))))
    bstar = (alpha * b + rho * torch.sum(U - torch.mm(W, V), dim=1).reshape(b.size())) / (rho * N + alpha)
    return Wstar, bstar


def relu_prox(a, b, gamma, d, N):
    x = (a + gamma * b) / (1 + gamma)
    y = torch.min(b, torch.zeros(d, N))

    val = torch.where(a + gamma * b < 0, y, torch.zeros(d, N))
    val = torch.where(
        ((a + gamma * b >= 0) & (b >= 0)) | ((a * (gamma - np.sqrt(gamma * (gamma + 1))) <= gamma * b) & (b < 0)), x,
        val)
    val = torch.where((-a <= gamma * b) & (gamma * b <= a * (gamma - np.sqrt(gamma * (gamma + 1)))), b, val)
    return val


def block_update(Wn, bn, Wn_1, bn_1, Un, Vn_1, Un_1, Vn_2, dn_1, alpha, gamma, rho, dim):
    # update W(n) and b(n)
    Wn, bn = updateWb(Un, Vn_1, Wn, bn, alpha, rho)
    # update V(n-1)
    Vn_1 = updateV(Un_1, Un, Wn, bn, rho, gamma)
    # update U(n-1)
    Un_1 = relu_prox(Vn_1, (rho * torch.addmm(bn_1.repeat(1, dim), Wn_1, Vn_2) +
                            alpha * Un_1) / (rho + alpha), (rho + alpha) / gamma, dn_1, dim)
    return Wn, bn, Vn_1, Un_1


class ModelBCD:

    def __init__(self, d0, d1, d2, d3, classes, gamma, alpha, rho):
        # Layer 1
        self.fc1 = nn.Linear(d0, d1)
        self.b1 = self.fc1.bias.reshape(d1, 1).data
        self.w1 = self.fc1.weight.data
        # Layer 2
        self.fc2 = nn.Linear(d1, d2)
        self.b2 = self.fc2.bias.reshape(d2, 1).data
        self.w2 = self.fc2.weight.data
        # Layer 3
        self.fc3 = nn.Linear(d2, d3)
        self.b3 = self.fc3.bias.reshape(d3, 1).data
        self.w3 = self.fc3.weight.data
        # Layer 4
        self.fc4 = nn.Linear(d3, classes)
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

    def init_aux_params(self, x):
        n_samples = x.size()[1]
        self.U1, self.V1 = feed_forward(self.w1, self.b1, x, n_samples)
        self.U2, self.V2 = feed_forward(self.w2, self.b2, self.V1, n_samples)
        self.U3, self.V3 = feed_forward(self.w3, self.b3, self.V2, n_samples)
        self.U4 = torch.addmm(self.b4.repeat(1, n_samples), self.w4, self.V3)
        self.V4 = self.U4

    def forward(self, x):
        n_samples = x.size()[1]
        _, a1 = feed_forward(self.w1, self.b1, x, n_samples)
        _, a2 = feed_forward(self.w2, self.b2, a1, n_samples)
        _, a3 = feed_forward(self.w3, self.b3, a2, n_samples)
        output = torch.addmm(self.b4.repeat(1, n_samples), self.w4, a3)
        return output

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
        self.w1, self.b1 = updateWb(self.U1, x, self.w1, self.b1, self.alpha, self.rho)
