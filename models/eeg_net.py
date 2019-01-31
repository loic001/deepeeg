import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class EEGNet(nn.Module):
    def __init__(self, input_time_length, chan):
        super(EEGNet, self).__init__()
        self.input_time_length = input_time_length
        self.chan = chan

        self.conv1 = nn.Conv2d(1, 16, (self.input_time_length, chan), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(4*2*7, 1)


    def forward(self, x):
        x = _transpose_time_to_spat(x)

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        x = x.view(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x

def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
