import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class feature_encoder(nn.Module):

    def __init__(self, num_layers, n_classes=1, bn=False):
        super(feature_encoder, self).__init__()

        self.fc_layer = self._make_layer(num_layers, bn=bn)
        self.linear = nn.Linear(num_layers[-1], n_classes)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, num_layers, bn=False):
        layers = []
        for i in range(len(num_layers)-1):
            layers.append(nn.Linear(num_layers[i], num_layers[i+1]))
            if bn:
                layers.append(nn.BatchNorm1d(num_layers[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5))    # added at 18/12/03
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1_block(x)
        x = self.fc_layer(x)
        x = self.linear(x)
        x = x.view(-1)
        return x

class feature_encoder_bn(nn.Module):

    def __init__(self, num_layers, n_classes=1, bn=False):
        super(feature_encoder_bn, self).__init__()

        self.fc_layer = self._make_layer(num_layers, bn=bn)
        self.linear = nn.Linear(num_layers[-1], n_classes)
        self.bn = nn.BatchNorm1d(n_classes)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, num_layers, bn=False):
        layers = []
        for i in range(len(num_layers)-1):
            layers.append(nn.Linear(num_layers[i], num_layers[i+1]))
            if bn:
                layers.append(nn.BatchNorm1d(num_layers[i+1]))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=0.5))    # added at 18/12/03
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1_block(x)
        x = self.fc_layer(x)
        x = self.linear(x)
        x = self.bn(x)
        x = x.view(-1)
        return x

class pre_encoder(nn.Module):

    def __init__(self, num_layers, n_classes=1):
        super(pre_encoder, self).__init__()

        self.fc_layer = self._make_layer(num_layers)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, num_layers):
        layers = []
        for i in range(len(num_layers)-1):
            layers.append(nn.BatchNorm1d(num_layers[i]))
            layers.append(nn.Linear(num_layers[i], num_layers[i+1]))
            # layers.append(nn.BatchNorm1d(num_layers[i+1]))
            # if i != len(num_layers)-2:
            #     layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=0.5))    # added at 12/03
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layer(x)
        return x


def Feature_encoder(num_layers=[256,512,512], bn=False):
    return feature_encoder(num_layers, n_classes=1, bn=bn)

def Feature_encoder_bn(num_layers=[256,512,512], bn=False):
    return feature_encoder_bn(num_layers, n_classes=1, bn=bn)

def Pre_encoder(num_layers):
    return pre_encoder(num_layers)