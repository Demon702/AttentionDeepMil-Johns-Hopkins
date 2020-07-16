import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from resnet import resnet18
from resnet_classifier import Resnet_Classifier
class Attention(nn.Module):
    def __init__(self , path = ""):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
 
        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        model_ft = Resnet_Classifier()
        # model_ft = models.resnet34(pretrained=True)
        if path!= "":
            model_ft.load_state_dict(torch.load(path))
        for param in model_ft.parameters():
            param.requires_grad = False
        self.feature_extractor_part1 = model_ft
        num_ftrs = model_ft.model_ft.fc.in_features
        # num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).    

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(3, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(num_ftrs, self.L),
            nn.Sigmoid(),
        )

        self.feature_extractor_part1.model_ft.fc  = self.feature_extractor_part2
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.gate = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.weight = nn.Linear(self.D , self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            # nn.ReLU()
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        # print(x.shape)
        # x = x[: , 1]
        # with torch.no_grad():
        H = self.feature_extractor_part1(x)
        # for param in self.feature_extractor_part1.parameters():
        #     print(param.requires_grad)
            # H = H[: , 1].unsqueeze(0)
            # H = torch.transpose(H , 0 , 1)
        # print(H.shape)
        # print(H.shape)
        # H = H.view(-1, 50 * 53 * 53)
        # H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        G = self.gate(H)     # N*K

        Gated_attention = self.weight(A *G) # N * K
        # Gated_attention = self.weight(A)
        # print(Gated_attention.shape)
        Gated_attention = torch.transpose(Gated_attention, 1, 0)  # KxN
        Weights = F.softmax(Gated_attention, dim=1)  # softmax over N
        # print(Weights)
        Y_prob = torch.mm(Weights , H)  # KxL
        Y_prob = self.classifier(Y_prob)
        # preds = torch.ge(Y_prob, 0.5)

        return Y_prob ,  Weights , H

    # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

    #     return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
