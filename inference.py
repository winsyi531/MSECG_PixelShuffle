import os, argparse
import torch
import numpy as np
from model.model import MSECG
from utils.dataloader import test_dataset, get_dataset_filelist
from tqdm import tqdm
import wfdb
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mamba_in_ch', type=int, default=128, help='channels for input of Mamba block')
    parser.add_argument('--n_layer', type=int, default=1, help='number of layers for Mamba block')
    parser.add_argument('--bidirectional', type=str, default='False', help='True or False for Bi-Directional Mamba')
    parser.add_argument('--pth_path', type=str, default='./model_pth/MSECG-best.pth') # input model checkpoint file
    parser.add_argument('--downsample_rate', type=int, default=10, help='how many times to down sample the signals')
    parser.add_argument('--input_testing_file', type=str, default='./dataset_index/test.txt')
    parser.add_argument('--noise_dir', type=str, default='/dataset/MIT-BIH_noise/')
    opt = parser.parse_args()

    ### set network ###
    model = MSECG(opt.mamba_in_ch, opt.n_layer, opt.bidirectional)
    
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
    test_loader = test_dataset(testing_indexes, opt)
    num1 = len(testing_indexes)

    for i in tqdm(range(num1), (f'Inference'), unit=' signal'):
        ns_signal, _, name = test_loader.load_data()
        ns_signal = ns_signal.cuda()
        
        sr_audio = model(ns_signal)
        sr_audio = torch.transpose(sr_audio, 1, 2).to(dtype=torch.float64).cpu().detach().numpy().squeeze()
        np.save(save_path+name+'_sr.npy', sr_audio)
    print('Finish inference process.')
