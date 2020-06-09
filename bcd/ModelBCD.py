import torch
import torch.nn as nn
from .utils import *

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
        self.V0 = x
        self.U1, self.V1 = feed_forward(self.w1, self.b1, self.V0, n_samples)
        self.U2, self.V2 = feed_forward(self.w2, self.b2, self.V1, n_samples)
        self.U3, self.V3 = feed_forward(self.w3, self.b3, self.V2, n_samples)
        self.U4 = torch.addmm(self.b4.repeat(1, n_samples), self.w4, self.V3)
        self.V4 = self.U4 #as sigma_4 = Id
    

    def forward(self, x):
        n_samples = x.size()[1]
        V1 = feed_forward(self.w1, self.b1, x, n_samples)[1]
        V2 = feed_forward(self.w2, self.b2, V1, n_samples)[1]
        V3 = feed_forward(self.w3, self.b3, V2, n_samples)[1]
        output = torch.addmm(self.b4.repeat(1, n_samples), self.w4, V3)
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
        self.w1, self.b1 = updateWn(self.U1, x, self.w1, self.b1, self.alpha, self.rho)
