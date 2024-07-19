import os
import torch
import torch.utils.data as data
import numpy as np
import random as rd
import pandas as pd
import wfdb
from scipy.signal import resample
import scipy.signal as signal
from copy import deepcopy
rd.seed(0)

class PTB_XLDataset(data.Dataset):
    """
    dataloader for audio SR tasks
    """
    def __init__(self, training_indexes, opt):
        self.signals = training_indexes
        self.down_sample = opt.downsample_rate
        self.mix_csv = pd.read_csv(opt.noise_dir+'info.csv')
        # Get the filename column in info.csv as a list
        self.csv_filename = self.mix_csv.filename.to_list()
        # Initialize the self.noise dictionary with three noise arrays
        self.noise = {}
        noise_type = ['bw', 'em', 'ma']
        for noise in noise_type:
            noise_path = opt.noise_dir + noise
            noise_wave, _ = wfdb.rdsamp(noise_path)
            noise_wave = np.array(noise_wave, dtype=np.float32)
            #downsample the noise from 360Hz to 50Hz
            down_fs = (360//50)
            noise_down = noise_wave[::down_fs]
            self.noise[noise] = noise_down
        self.size = len(self.signals)

    def __getitem__(self, index):
        # Get only the number of signal and produce index
        filename = self.signals[index].split('/')[-1].split('_')[0]
        index_filename = self.csv_filename.index(int(filename))
        # Get noise type, noise channel, noise SNR from info.csv
        lr_noise_type = self.mix_csv.noise_type.to_list()[index_filename]
        noise_ch = self.mix_csv.noise_channel.to_list()[index_filename]
        noise_snr = self.mix_csv.SNR.to_list()[index_filename]
        # Read 500Hz HR signals
        hr_signal, meta = wfdb.rdsamp(self.signals[index])
        hr_signal = np.array(hr_signal, dtype=np.float32)
        hr_signal = butter_bandpass_filter(data=hr_signal, lowcut=1, highcut=45, fs=500, order=2)
        
        # adjust the length of signal to 5000 sampling points
        if hr_signal.shape[0] < 5000:
            hr_signal = np.pad(hr_signal, (0, (5000-sig.shape[0])), 'constant', constant_values=(0))
        elif hr_signal.shape[0] > 5000:
            hr_signal = hr_signal[:5000,:]
        assert hr_signal.shape[0] == 5000

        # Down-sample the hr_signal to 50Hz
        down_fs = 10
        ds_signal = hr_signal[::down_fs] # (500, 12)
        assert ds_signal.shape[0] == 500, f'Downsampled signal does not match the downsampling result!!!'
        
        #sample the noise segment start point
        ns_signal = deepcopy(ds_signal) # (500, 12)
        if lr_noise_type != 'norm':
            if lr_noise_type == 'bw':
                start = rd.randint(0, self.noise['bw'].shape[0]-500)
                noise_segment = self.noise['bw'][start:start+500]
            elif lr_noise_type == 'em':
                start = rd.randint(0, self.noise['em'].shape[0]-500)
                noise_segment = self.noise['em'][start:start+500]
            elif lr_noise_type == 'ma':
                start = rd.randint(0, self.noise['ma'].shape[0]-500)
                noise_segment = self.noise['ma'][start:start+500]

            # add noise
            # Load channel of noise added to the signal
            for i in range(12):
                y_clean_pw = np.dot(ds_signal[:, i] ,ds_signal[:, i])
                y_noise_pw = np.dot(noise_segment[:, noise_ch], noise_segment[:, noise_ch])
                scalar = np.sqrt(y_clean_pw/((10.0**(noise_snr/10.0))*y_noise_pw))
                noise = scalar * noise_segment[:, noise_ch]
                y_noisy = ds_signal[:, i] + noise
                ns_signal[:, i] = y_noisy

        ns_signal = torch.transpose(torch.FloatTensor(ns_signal), 0, 1)
        hr_signal = torch.transpose(torch.FloatTensor(hr_signal), 0, 1)

        return ns_signal, hr_signal

    def __len__(self):
        return self.size


def get_loader(training_indexes, opt, shuffle=True, num_workers=12, pin_memory=True):

    dataset = PTB_XLDataset(training_indexes, opt)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, validation_indexes, opt):
        self.signals = validation_indexes
        self.down_sample = opt.downsample_rate
        self.mix_csv = pd.read_csv(opt.noise_dir+'info.csv')
        # Get the filename column in info.csv as a list
        self.csv_filename = self.mix_csv.filename.to_list()
        # Initialize the self.noise dictionary with three noise arrays
        self.noise = {}
        noise_type = ['bw', 'em', 'ma']
        for noise in noise_type:
            noise_path = opt.noise_dir + noise
            noise_wave, _ = wfdb.rdsamp(noise_path)
            noise_wave = np.array(noise_wave, dtype=np.float32)
            #downsample the noise from 360Hz to 50Hz
            down_fs = (360//50)
            noise_down = noise_wave[::down_fs]
            self.noise[noise] = noise_down
        self.size = len(self.signals)
        self.index = 0

    def load_data(self):
        # Get only the number of signal and produce index
        filename = self.signals[self.index].split('/')[-1].split('_')[0]
        index_filename = self.csv_filename.index(int(filename))
        # Get noise type, noise channel, noise SNR from info.csv
        lr_noise_type = self.mix_csv.noise_type.to_list()[index_filename]
        noise_ch = self.mix_csv.noise_channel.to_list()[index_filename]
        noise_snr = self.mix_csv.SNR.to_list()[index_filename]
        # Read 500Hz HR signals
        hr_signal, meta = wfdb.rdsamp(self.signals[self.index])
        hr_signal = np.array(hr_signal, dtype=np.float32)
        hr_signal = butter_bandpass_filter(data=hr_signal, lowcut=1, highcut=45, fs=500, order=2)
        
        # adjust the length of signal to 5000 sampling points
        if hr_signal.shape[0] < 5000:
            hr_signal = np.pad(hr_signal, (0, (5000-hr_signal.shape[0])), 'constant', constant_values=(0))
        elif hr_signal.shape[0] > 5000:
            hr_signal = hr_signal[:5000,:]
        assert hr_signal.shape[0] == 5000
        
        # Down-sample the hr_signal to 50Hz
        down_fs = 10
        ds_signal = hr_signal[::down_fs] # (500, 12)
        assert ds_signal.shape[0] == 500, f'Downsampled signal does not match the downsampling result!!!'

        #sample the noise segment start point
        ns_signal = deepcopy(ds_signal) # (500, 12)
        if lr_noise_type != 'norm':
            if lr_noise_type == 'bw':
                start = rd.randint(0, self.noise['bw'].shape[0]-500)
                noise_segment = self.noise['bw'][start:start+500]
            elif lr_noise_type == 'em':
                start = rd.randint(0, self.noise['em'].shape[0]-500)
                noise_segment = self.noise['em'][start:start+500]
            elif lr_noise_type == 'ma':
                start = rd.randint(0, self.noise['ma'].shape[0]-500)
                noise_segment = self.noise['ma'][start:start+500]

            # add noise
            # Load channel of noise added to the signal
            for i in range(12):
                y_clean_pw = np.dot(ds_signal[:, i] ,ds_signal[:, i])
                y_noise_pw = np.dot(noise_segment[:, noise_ch], noise_segment[:, noise_ch])
                scalar = np.sqrt(y_clean_pw/((10.0**(noise_snr/10.0))*y_noise_pw))
                noise = scalar * noise_segment[:, noise_ch]
                y_noisy = ds_signal[:, i] + noise
                ns_signal[:, i] = y_noisy
        
        ns_signal = torch.transpose(torch.FloatTensor(ns_signal), 0, 1).unsqueeze(0)
        hr_signal = torch.transpose(torch.FloatTensor(hr_signal), 0, 1).unsqueeze(0)

        self.index += 1

        return ns_signal, hr_signal, filename

# Get index from 'train.txt', 'val.txt', and 'test.txt'
def get_dataset_filelist(opt, mode):
    if mode == 'train':
        with open(opt.input_training_file, 'r', encoding='utf-8') as fi:
            training_indexes = [x for x in fi.read().split('\n') if len(x) > 0]
        with open(opt.input_validation_file, 'r', encoding='utf-8') as fi:
            validation_indexes = [x for x in fi.read().split('\n') if len(x) > 0]
        return training_indexes, validation_indexes
    elif mode == 'test':
        with open(opt.input_testing_file, 'r', encoding='utf-8') as fi:
            testing_indexes = [x for x in fi.read().split('\n') if len(x) > 0]
        return testing_indexes

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    normal_low = lowcut / nyquist
    normal_high = highcut / nyquist
    b, a = signal.butter(order, [normal_low, normal_high], btype='band', analog=False)
    
    frame, channel = data.shape
    for c in range(channel):
        data[:, c] = signal.filtfilt(b, a, data[:, c])
    return data
