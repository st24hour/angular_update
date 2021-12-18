'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Loss_prediction_module(nn.Module):
    def __init__(self, channels):
        super(Loss_prediction_module, self).__init__()
        self.fc_1 = torch.nn.Linear(channels[0], 128)
        self.fc_2 = torch.nn.Linear(channels[1], 128)
        self.fc_3 = torch.nn.Linear(channels[2], 128)
        self.fc_4 = torch.nn.Linear(channels[3], 128)
        self.fc = torch.nn.Linear(512, 1)


    def forward(self, features):
        feature_1 = features[0]
        feature_2 = features[1]
        feature_3 = features[2]
        feature_4 = features[3]
        # b, _, _, _ = feature_1.size()
        feature_1 = F.relu(self.fc_1(F.avg_pool2d(feature_1, 32).squeeze()))
        feature_2 = F.relu(self.fc_2(F.avg_pool2d(feature_2, 16).squeeze()))
        feature_3 = F.relu(self.fc_3(F.avg_pool2d(feature_3, 8).squeeze()))
        feature_4 = F.relu(self.fc_4(F.avg_pool2d(feature_4, 4).squeeze()))

        features = torch.cat((feature_1, feature_2, feature_3, feature_4), 1)

        features = self.fc(features)

        return features

def loss_pred_module(channels=[64, 128, 256, 512]):
    return Loss_prediction_module(channels)

# dense: 96, 192, 384, 384
