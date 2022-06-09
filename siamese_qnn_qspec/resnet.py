import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import core_qnn
from core_qnn import quaternion_layers as qnn
from core_qnn import quaternion_ops as qop


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuaternionConv(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(),
            qnn.QuaternionConv(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        return self.ReLU(x1+x)


class DownBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBasicBlock, self).__init__()
        self.conv = nn.Sequential(
            qnn.QuaternionConv(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            qnn.QuaternionConv(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.ReLU = nn.ReLU()
        self.down = qnn.QuaternionConv(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down(x)
        return self.ReLU(x1+x2)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = qnn.QuaternionConv(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = qnn.QuaternionConv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = qnn.QuaternionConv(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.max = nn.MaxPool2d(3, 2)
        self.resnet = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            DownBasicBlock(64, 128),#32,32
            BasicBlock(128, 128),
            DownBasicBlock(128, 256),#16,16
            BasicBlock(256, 256),
            # DownBasicBlock(256, 512),#8,8
            # BasicBlock(512, 512),
        )
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x0 = self.layer1(x)
        x0 = self.ReLU(x0)
        x0 = self.layer2(x0)
        x0 = self.ReLU(x0)
        x0 = self.layer3(x0)
        x0 = self.ReLU(x0)
        # x0 = self.max(x0)
        x1 = self.resnet(x0)
        x2 = self.GAP(x1)
        x2 = x2.view(-1, 256)
        x3 = self.fc(x2)
        return x3


res = ResNet()
print(sum(param.numel() for param in res.parameters()))