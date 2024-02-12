import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=63, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2944, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.init_fc1 = nn.Linear(33, 64)
        self.init_fc2 = nn.Linear(64, 128)

        self.act = nn.ReLU()
        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(64)
        self.bn4 = nn.InstanceNorm2d(32)
        self.bn5 = nn.InstanceNorm2d(16)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.concatenate((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.drop1(self.fc1(x)))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x
    
class CNN_nopos(nn.Module):
    def __init__(self):
        super(CNN_nopos, self).__init__()
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        # self.pre_conv= nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2880, 64)
        # self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.init_fc1 = nn.Linear(3, 32)
        self.init_fc2 = nn.Linear(32, 64)

        self.act = nn.ReLU()
        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(64)
        self.bn3 = nn.InstanceNorm2d(64)
        self.bn4 = nn.InstanceNorm2d(32)
        self.bn5 = nn.InstanceNorm2d(16)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
      

    def forward(self, patches, init):

        x = self.act((self.conv1(patches)))
        # print(x.shape)
        x = self.act((self.conv2(x)))
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x_in = self.act(self.init_fc1(init))
        # print(x_in.shape)
        x_in = self.act(self.init_fc2(x_in))
        # print(x_in.shape)
        x = torch.concatenate((x, x_in), dim=-1)
        # print(x.shape)
        x = self.act(self.drop1(self.fc1(x)))
        # print(x.shape)
        x = self.act(self.drop2(self.fc2(x)))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = torch.tanh(x)
        # print(x.shape)
        
        return x


if __name__ == '__main__':
    dumm = torch.rand(4,3,12,26)
    dumm_norm = torch.rand(4,3)
    # model = CNN()
    # out = model(dumm, dumm_norm)
    model = CNN_nopos()
    out = model(dumm, dumm_norm)
    print(out.shape)

