import torch
import torch.nn as nn
import numpy as np
import os
import core_qnn
from core_qnn import quaternion_layers as qnn
from core_qnn import quaternion_ops as qop


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            qnn.QuaternionConv(in_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = qnn.QuaternionTransposeConv(in_channels, out_channels, stride=2, kernel_size=2, padding=0, bias=False)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            qnn.QuaternionConv(64, 64, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            qnn.QuaternionConv(64, 64, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            qnn.QuaternionConv(128, 128, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            qnn.QuaternionConv(128, 128, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            qnn.QuaternionConv(256, 256, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            qnn.QuaternionConv(256, 256, stride=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
            )
        self.netEntry = qnn.QuaternionConv(4, 64, stride=1, kernel_size=3, padding=1, bias=False)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.netExit = qnn.QuaternionConv(64, 4, stride=1, kernel_size=3, padding=1, bias=False)
        self.RelU = nn.ReLU()
    def forward(self, x):

        x1 = self.netEntry(x)
        x1 = self.RelU(x1)

        x1 = self.conv1(x1)

        x2 = self.down1(x1)
        x2 = self.RelU(x2)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x3 = self.RelU(x3)
        x3 = self.conv3(x3)

        x4 = self.up1(x3) + x2
        x4 = self.RelU(x4)
        x4 = self.conv2(x4)

        x5 = self.up2(x4) + x1
        x5 = self.RelU(x5)
        x5 = self.conv1(x5)

        x6 = self.netExit(x5)
        x6 = self.RelU(x6)
        return x6

unet = UNet()
print(sum(param.numel() for param in unet.parameters()))

