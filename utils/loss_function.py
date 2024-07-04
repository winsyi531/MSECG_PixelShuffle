import os
import torch
import torch.nn.functional as F
import numpy as np

'''
This is a file containing several loss functions
'''

def mag_pha_stft(y, n_fft=256, hop_size=128, win_size=256, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).cuda()
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    # Get real and imaginary parts
    real_part = stft_spec.real
    imag_part = stft_spec.imag

    # compute magnitude and phase
    mag = torch.sqrt(real_part.pow(2)+imag_part.pow(2)+1e-10)
    pha = torch.atan2(real_part+1e-10, imag_part+1e-10)

    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def MSE_LOSS(pred, gt):
    """
    input:
        pred: output from network
        mask: ground truth
    output:
        loss value (MSE)
    """
    mse = F.mse_loss(pred, gt)
    
    return mse

def MAG_MSE_Loss(pred, gt, compress_factor=1.0):
    """
    input:
        pred: output from network
        mask: ground truth
    output:
        loss value (STFT_MSE)
    """
    assert pred.shape == gt.shape, f'The shapes of predicted and GT signals calculating STFT loss do not match!!!'
    
    total_loss = 0
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            mag_pred, pha_pred, com_pred = mag_pha_stft(pred[b, c, :].squeeze(), compress_factor=1.0)
            mag_gt, pha_gt, com_gt = mag_pha_stft(gt[b, c, :].squeeze(), compress_factor=1.0)

            # Compute MSE loss
            mag_mse_loss = F.mse_loss(mag_pred, mag_gt)
            total_loss += mag_mse_loss

    return total_loss / (pred.shape[0]*pred.shape[1])

def COM_MSE_Loss(pred, gt, compress_factor=1.0):
    """
    input:
        pred: output from network
        mask: ground truth
    output:
        loss value (STFT_MSE)
    """
    assert pred.shape == gt.shape, f'The shapes of predicted and GT signals calculating STFT loss do not match!!!'
    
    total_loss = 0
    for b in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            mag_pred, pha_pred, com_pred = mag_pha_stft(pred[b, c, :].squeeze(), compress_factor=1.0)
            mag_gt, pha_gt, com_gt = mag_pha_stft(gt[b, c, :].squeeze(), compress_factor=1.0)

            # Compute MSE loss
            mag_mse_loss = F.mse_loss(com_pred, com_gt)
            total_loss += mag_mse_loss

    return total_loss / (pred.shape[0]*pred.shape[1])









