"""
This python file includes every module adopted in WCNet.
Any module is free to add in this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

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

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. 
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, C, L = x.shape
        up_C = C // self.upscale_factor
        up_L = L * self.upscale_factor

        # This pixel shuffle might be wrong
        x = x.contiguous().view(B, up_C, self.upscale_factor, L)  # [1, 12, 10, 500]
        x = x.permute(0, 1, 3, 2).contiguous()  # [1, 12, 500, 10]
        x = x.view(B, up_C, up_L)  # [1, 12, 5000]

        # This pixel shuffle should be right
        #x = x.view(B, up_C, self.upscale_factor, L)  # [1, 12, 10, 500]
        #x = x.view(B, up_C, up_L)  # [1, 12, 5000]
        '''
        up_x = torch.zeros((B, up_C, up_L)).cuda()
        for b in range(B):
            for c in range(up_C):
                hold_x = torch.tensor(()).cuda()
                for u in range(self.upscale_factor):
                    hold_x = torch.cat((hold_x, x[b, c*self.upscale_factor+u, :]), -1)
                up_x[b, c, :] = hold_x.unsqueeze(0)
        '''
        return x

class MambaBlock(nn.Module):
    def __init__(self, in_channels=64, n_layer=1, bidirectional='False'):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        self.backward_blocks = None
        if bidirectional == 'True':
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=2),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        
        return residual