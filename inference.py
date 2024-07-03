import os, argparse
import torch
import numpy as np
from model.model import SRECG
from utils.dataloader import test_dataset, get_dataset_filelist
from tqdm import tqdm
import wfdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='./model_pth/SRECG-best.pth') # input model checkpoint file
    parser.add_argument('--downsample_rate', type=int, default=10, help='how many times to down sample the signalse')
    parser.add_argument('--input_testing_file', type=str, default='./dataset_index/test.txt')
    opt = parser.parse_args()

    ### set network ###
    model = SRECG()
    
    ### input model checkpoint file ###
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    ### save_path ###
    save_path = './sr_signal/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mode = 'test'
    testing_indexes = get_dataset_filelist(opt, mode)
    test_loader = test_dataset(testing_indexes, opt.downsample_rate)
    num1 = len(testing_indexes)

    for i in tqdm(range(num1), (f'Inference'), unit=' signal'):
        ds_audio, _, name = test_loader.load_data()
        ds_audio = ds_audio.cuda()
        
        sr_audio = model(ds_audio)
        sr_audio = torch.transpose(sr_audio, 1, 2).to(dtype=torch.float64).cpu().detach().numpy().squeeze()
        np.save(save_path+name+'.npy', sr_audio)
    print('Finish inference process.')
        #wfdb.wrsamp(
        #    record_name = save_path + name,
        #    fs = 500,
        #    units = ['mV'] * sr_audio.shape[1],
        #    sig_name = [f'ECG{i+1}' for i in range(sr_audio.shape[1])],
        #    p_signal = sr_audio
        #)
