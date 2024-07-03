"""
This python file includes every module adopted in WCNet.
Any module is free to add in this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(in_ch),
            nn.LeakyReLU(),
            nn.Conv1d(in_ch, in_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(in_ch)
        )
    
    def forward(self, x):
        res = x
        x1 = self.conv_block(x)
        out = res + x1
        return out

class UpConv_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv_5, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=16,
            stride=5,
            padding=6,
            output_padding=1
        )
        self.activation = nn.LeakyReLU()
        
    def forward(self, input):
        out = self.conv_transpose(input)
        out = self.activation(out)
        return out

class UpConv_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv_2, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=16,
            stride=2,
            padding=7,
            output_padding=0
        )
        self.activation = nn.LeakyReLU()
        
    def forward(self, input):
        out = self.conv_transpose(input)
        out = self.activation(out)
        return out

