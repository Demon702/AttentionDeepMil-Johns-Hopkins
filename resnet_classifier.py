import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Resnet_Classifier(nn.Module):
    def __init__(self):
        super(Resnet_Classifier, self).__init__()
        self.model_ft = models.resnet34(pretrained=True)
        num_ftrs = self.model_ft.fc.out_features
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(num_ftrs , 2)

    def forward(self, x):
        # x = x.squeeze(0)
        x = self.model_ft(x)
        prob = F.softmax(self.linear(x))

        return prob


    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
