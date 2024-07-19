import os
import numpy as np
import torch
import torch.nn.functional as F
import wfdb
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from utils.dataloader import butter_bandpass_filter

def calculate_snr(gt_signal, pred_signal):
    noise = gt_signal - pred_signal
    signal_power = np.sum(pred_signal**2)
    noise_power = np.sum(noise**2)
    snr = 10 * np.log10(signal_power/noise_power)
    return snr

def calculate_psnr(gt_signal, pred_signal):
    mse = np.mean((gt_signal - pred_signal)**2) + 1e-10
    max_signal = np.max(pred_signal)
    psnr = 10 * np.log10((max_signal**2)/mse)
    return psnr

def calculate_mse(gt_signal, pred_signal):
    mse = np.mean((pred_signal-gt_signal)**2) + 1e-10
    return mse

if __name__ == '__main__':
    ### path of testing signals and ground truth ###
    gt_index = './dataset_index/test.txt'
    pd_path = './sr_signal/'
    txt_path_mse  = './ev_mse.txt'
    txt_path_rmse = './ev_rmse.txt'
    txt_path_ssim = './ev_ssim.txt'
    txt_path_snr  = './ev_snr.txt'
    txt_path_psnr = './ev_psnr.txt'
    ###############################################
    
    ### find image with 'png' image and sorting ###
    with open(gt_index, 'r', encoding='utf-8') as fi:
        gt_name = [x for x in fi.read().split('\n') if len(x) > 0]
    pd_name = [_ for _ in os.listdir(pd_path) if _.endswith('_sr.npy')]
    gt_name.sort()
    pd_name.sort()
    
    gt_num = len(gt_name)
    pd_num = len(pd_name)
    assert gt_num == pd_num, f'The numbers of predicted images and ground truth signals do not match!!!'
    ################################################
    
    average_mse   = 0
    average_rmse  = 0
    average_ssim  = 0
    average_snr   = 0
    average_psnr  = 0

    ### calculate the average of dice, recall, precision value of all testing images ###
    for i, name in enumerate(tqdm(gt_name, (f'Evaluation'),unit=' signal')):
        gt_sig, _ = wfdb.rdsamp(name)
        gt_sig = butter_bandpass_filter(data=gt_sig, lowcut=1, highcut=45, fs=500, order=2)
        pd_sig = np.load(pd_path+pd_name[i])
        gt_sig, pd_sig = gt_sig.flatten(), pd_sig.flatten()
        assert len(gt_sig) == len(pd_sig), f'The lengths between ground truth and predicted signals do not match!!!'
    
        mse_score = calculate_mse(gt_sig, pd_sig)
        rmse_score = calculate_mse(gt_sig, pd_sig) ** 0.5
        ssim_score = ssim(gt_sig, pd_sig, data_range=gt_sig.max()-gt_sig.min())
        snr_score = calculate_snr(gt_sig, pd_sig)
        psnr_score = calculate_psnr(gt_sig, pd_sig)
        
        average_mse += mse_score
        average_rmse += rmse_score
        average_ssim += ssim_score
        average_snr += snr_score
        average_psnr += psnr_score
        
    
    ### record three value in five txt files ###
    average_mse  /= pd_num
    average_rmse /= pd_num
    average_ssim /= pd_num
    average_snr /= pd_num
    average_psnr /= pd_num

    f = open(txt_path_mse, 'a')
    f.write('{:.6f}\n'.format(average_mse))
    f.close()
    print("Total Average MSE Score: {:.6f}".format(average_mse))
    
    f = open(txt_path_rmse, 'a')
    f.write('{:.6f}\n'.format(average_rmse))
    f.close()
    print("Total Average RMSE Score: {:.6f}".format(average_rmse))
    
    f = open(txt_path_ssim, 'a')
    f.write('{:.6f}\n'.format(average_ssim))
    f.close()
    print("Total Average SSIM Score: {:.6f}".format(average_ssim))
    
    f = open(txt_path_snr, 'a')
    f.write('{:.6f}\n'.format(average_snr))
    f.close()
    print("Total Average SNR Score: {:.6f}".format(average_snr))
    
    f = open(txt_path_psnr, 'a')
    f.write('{:.6f}\n'.format(average_psnr))
    f.close()
    print("Total Average PSNR Score: {:.6f}".format(average_psnr))
