import os
import torch
import torch.utils.data as data
import numpy as np
import random
import pandas as pd
import wfdb
from scipy.signal import resample

class PTB_XLDataset(data.Dataset):
    """
    dataloader for audio SR tasks
    """
    def __init__(self, training_indexes, down_sample):
        self.signals = training_indexes
        self.down_sample = down_sample
        self.size = len(self.signals)

    def __getitem__(self, index):
        signal, meta = wfdb.rdsamp(self.signals[index])
        signal = np.array(signal, dtype=np.float32)
        
        # adjust the length of signal to 5000 sampling points
        if signal.shape[0] < 5000:
            signal = np.pad(signal, (0, (5000-sig.shape[0])), 'constant', constant_values=(0))
        elif signal.shape[0] > 5000:
            signal = signal[0:5000,:]
        assert signal.shape[0] == 5000

        origin_fs = signal.shape[0]
        down_fs = origin_fs // self.down_sample # 5000 // 10 = 500
        down_sig = resample(signal, down_fs, axis=0)
        down_sig = torch.transpose(torch.FloatTensor(down_sig), 0, 1)
        signal = torch.transpose(torch.FloatTensor(signal), 0, 1)

        return down_sig, signal

    def __len__(self):
        return self.size


def get_loader(training_indexes, down_sample, batchsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = PTB_XLDataset(training_indexes, down_sample)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, validation_indexes, down_sample):
        self.signals = validation_indexes
        self.down_sample = down_sample
        self.size = len(self.signals)
        self.index = 0

    def load_data(self):
        signal, meta = wfdb.rdsamp(self.signals[self.index])
        signal = np.array(signal, dtype=np.float32)
        
        # adjust the length of signal to 5000 sampling points
        if signal.shape[0] < 5000:
            signal = np.pad(signal, (0, (5000-sig.shape[0])), 'constant', constant_values=(0))
        elif signal.shape[0] > 5000:
            signal = signal[0:5000,:]
        assert signal.shape[0] == 5000
        
        origin_fs = signal.shape[0]
        down_fs = origin_fs // self.down_sample # 5000 // 10 = 500
        down_sig = resample(signal, down_fs, axis=0)
        down_sig = torch.transpose(torch.FloatTensor(down_sig), 0, 1).unsqueeze(0)
        signal = torch.transpose(torch.FloatTensor(signal), 0, 1).unsqueeze(0)

        name = self.signals[self.index].split('/')[-1]
        self.index += 1

        return down_sig, signal, name

### Get index from 'train.txt', 'val.txt', and 'test.txt'
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
############################################
