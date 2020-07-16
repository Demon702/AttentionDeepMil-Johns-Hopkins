import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from resnet import resnet34
class Resnet_Classifier(nn.Module):
    def __init__(self):
        super(Resnet_Classifier, self).__init__()
        self.model_ft = resnet34(pretrained=True)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        # self.sigmoid = nn.Sigmoid()
        self.model_ft.fc = nn.Linear(num_ftrs , 2)
        # self.conv2d = torch.nn.Conv2d(3, 3, 4, stride = 2)

    def forward(self, x):
        # x = x.squeeze(0
        # with torch.no_grad(): 
        # x = F.relu(self.conv2d(x))
        # print(x.size())
        x = self.model_ft(x)
        # x = self.linear(x)
        # prob = self.linear(x)

        return x


    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
