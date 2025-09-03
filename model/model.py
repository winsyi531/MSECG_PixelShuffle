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

class MSECG(nn.Module):
    def __init__(self, mamba_in_ch=64, n_layer=1, bidirectional='False'):
        super(SRECG, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=mamba_in_ch, kernel_size=15, padding=7),
            nn.LeakyReLU()
        )
        self.mamba = MambaBlock(in_channels=mamba_in_ch, n_layer=n_layer, bidirectional=bidirectional)
        if bidirectional == 'True':
            self.conv_merge = nn.Sequential(
                nn.Conv1d(in_channels=mamba_in_ch*2, out_channels=mamba_in_ch, kernel_size=15, padding=7),
                nn.BatchNorm1d(mamba_in_ch)
            )
        elif bidirectional == 'False':
            self.conv_merge = nn.Sequential(
                nn.Conv1d(in_channels=mamba_in_ch, out_channels=mamba_in_ch, kernel_size=15, padding=7),
                nn.BatchNorm1d(mamba_in_ch)
            )

        self.conv_ps1 = nn.Conv1d(in_channels=mamba_in_ch, out_channels=120, kernel_size=15, padding=7)
        self.ps1 = PixelShuffle1D(10)

    def forward(self, x):
        x_in = self.conv_in(x)
        x_in_hold = x_in
        x_in = x_in.permute(0, 2, 1) 
        x_in = self.mamba(x_in).permute(0, 2, 1)
        x_merge = self.conv_merge(x_in)
        x_merge = x_merge + x_in_hold
        x_merge = self.conv_ps1(x_merge)
        out = self.ps1(x_merge) + F.interpolate(x, scale_factor=10, mode='linear', recompute_scale_factor=False)
        
        return out

if __name__ == '__main__':
    model = MSECG().cuda()
    input_tensor = torch.randn(1, 12, 500).cuda()

    prediction1 = model(input_tensor)
    print(prediction1.size())
