"""
This python file is the main architecture of WCNet.
self.unet is the first CAU-Net.
Dimension behind feature map (C, H, W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .parts import *

class SRECG(nn.Module):
    def __init__(self):
        super(SRECG, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=64, kernel_size=15, padding=7),
            nn.LeakyReLU()
        )
        self.ResNet = nn.ModuleList([ResBlock(64) for i in range(16)])
        self.conv_merge = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64)
        )
        self.up1 = UpConv_5(64, 320)
        self.up2 = UpConv_2(320, 128)
        self.conv_out = nn.Conv1d(in_channels=128, out_channels=12, kernel_size=15, padding=7)

    def forward(self, x):
        x_in = self.conv_in(x)
        x_in_hold = x_in
        for block in self.ResNet:
            x_in = block(x_in)
        x_merge = self.conv_merge(x_in)
        x_merge = x_merge + x_in_hold
        x_up_5 = self.up1(x_merge)
        x_up_10 = self.up2(x_up_5)
        out = self.conv_out(x_up_10)

        return out

if __name__ == '__main__':
    model = SRECG().cuda()
    input_tensor = torch.randn(1, 12, 5000).cuda()

    prediction1 = model(input_tensor)
    print(prediction1.size())
