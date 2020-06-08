import torch
import torch.nn as nn
import numpy as np

def feed_forward(Wn, bn, Vn_1 ,n_samples):
    Un = torch.addmm(bn.repeat(1, n_samples), Wn, Vn_1) 
    Vn = nn.ReLU()(Un)
    return Un, Vn


def updateVn(Un, Un1, Wn1, bn1, rho, gamma):
    d = Wn1.size()[1]
    I = torch.eye(d)
    Vn = nn.ReLU()(Un)
    col_Un1 = Un1.size()[1]
    Vs = torch.mm(torch.inverse(rho * (torch.mm(torch.t(Wn1), Wn1)) + gamma * I),
                     rho * torch.mm(torch.t(Wn1), Un1 - bn1.repeat(1, col_Un1)) + gamma * Vn)
    return Vs


def updateWn(Un, Vn_1, Wn, bn, alpha, rho):
    d, N = Vn_1.size()
    I = torch.eye(d)
    col_Un = Un.size()[1]
    Ws = torch.mm(alpha * Wn + rho * torch.mm(Un - bn.repeat(1, col_Un), torch.t(Vn_1)),
                     torch.inverse(alpha * I + rho * (torch.mm(Vn_1, torch.t(Vn_1)))))
    bs = (alpha * bn + rho * torch.sum(Un - torch.mm(Wn, Vn_1), dim=1).reshape(bn.size())) / (rho * N + alpha)
    return Ws, bs


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
    Wn, bn = updateWn(Un, Vn_1, Wn, bn, alpha, rho)
    # update V(n-1)
    Vn_1 = updateVn(Un_1, Un, Wn, bn, rho, gamma)
    # update U(n-1)
    Un_1 = relu_prox(Vn_1, (rho * torch.addmm(bn_1.repeat(1, dim), Wn_1, Vn_2) +
                            alpha * Un_1) / (rho + alpha), (rho + alpha) / gamma, dn_1, dim)
    return Wn, bn, Vn_1, Un_1

