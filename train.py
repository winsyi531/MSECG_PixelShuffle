### import library ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from model.model import SRECG
from utils.dataloader import get_loader, test_dataset, get_dataset_filelist
from utils.generate_index import generate
from utils.loss_function import MSE_LOSS, MAG_MSE_Loss, COM_MSE_Loss
import numpy as np
from tqdm import tqdm
from torchinfo import summary
######################

### construct loss function for training ###


def structure_loss(pred, gt):
    """
    input:
        pred: output from network
        mask: ground truth
    output:
        loss value (mse+STFT_MSE)
    """
    loss_com = COM_MSE_Loss(pred, gt)
    loss_mse = MSE_LOSS(pred, gt)
    
    score = loss_mse + loss_com
    
    return score
############################################

### validation process during training ###
def test(opt, model, val_index, epoch):
    """
    input:
        opt: parse arguments
        model: network
        val_index: index for validation
        epoch: current epoch
    output:
        validation Dice performance
    """
    model.eval()
    test_loader = test_dataset(val_index, opt.downsample_rate)
    num1 = len(val_index)
    MSE = 0.0
    for i in tqdm(range(num1), (f'Epoch[{epoch}/{opt.epoch}]'), unit=' audio'):
        ds_audio, gt_audio, _ = test_loader.load_data()
        ds_audio = ds_audio.cuda()
        gt_audio = np.array(gt_audio, np.float32)

        sr_audio = model(ds_audio)
        sr_audio = sr_audio.cpu().detach().numpy()
        sr_audio = sr_audio.flatten()
        gt_audio = gt_audio.flatten()
        mse = ((sr_audio-gt_audio)**2).sum() / len(sr_audio)
        MSE = MSE + mse

    return MSE / num1 # output average MSE Score
############################################


### training process ###
def train(opt, train_loader, model, optimizer, epoch, validation_indexes):
    model.train()
    global best
    loop = tqdm(train_loader, unit=' batch')
    for i, pack in enumerate(loop):
        optimizer.zero_grad()
        # ---- data prepare ----
        ds_audio, gt_audio = pack
        ds_audio, gt_audio = ds_audio.cuda(), gt_audio.cuda()
        # ---- forward ----
        sr_audio = model(ds_audio)
        # ---- loss function ----         
        loss = structure_loss(sr_audio, gt_audio)
        # ---- backward ----
        loss.backward()
        optimizer.step()
        # ---- tqdm update ---- #
        loop.set_description(f'Epoch[{epoch}/{opt.epoch}]')
        loop.set_postfix(loss=loss.item())

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # choose the best model
    global dict_plot

    if epoch % 1 == 0:
        dataset = 'VCTK'
        dataset_mse = test(opt, model, validation_indexes, epoch) # use above test() function
        print(dataset, ':', dataset_mse)
        dict_plot[dataset].append(dataset_mse)
        
        meanmse = dataset_mse
        dict_plot['test'].append(meanmse)
        if (epoch==1) or (meanmse<best):
            best = meanmse # store best Dice Score
            torch.save(model.state_dict(), save_path+'SRECG-best.pth') # save best model
            print('########## Best Model Saved, MSE: {:.8f} ##########'.format(best))
####################################################################################################################################

if __name__ == '__main__':
    dict_plot = {'VCTK':[], 'test':[]}
    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')
    parser.add_argument('--downsample_rate', type=int,
                        default=10, help='how many times to down sample the signals')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/')
    parser.add_argument('--regenerate_index', default=False, help='whether want to generate the dataset')
    parser.add_argument('--dataset_dir', type=str, default='/dataset/PTB-XL/')
    parser.add_argument('--input_training_file', type=str, default='./dataset_index/train.txt')
    parser.add_argument('--input_validation_file', type=str, default='./dataset_index/val.txt')
    parser.add_argument('--input_testing_file', type=str, default='./dataset_index/test.txt')
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = SRECG().cuda()
    
    # ---- summary ---- #
    summary(model, (1, 12, 5000)) # show model architecture and parameters
    
    best = 0
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    train_fold = [1, 2, 3, 4, 5, 6, 7, 8]
    val_fold = [9]
    test_fold = [10]
    if opt.regenerate_index:
        generate(opt, train_fold, val_fold, test_fold)
    mode = 'train'
    training_indexes, validation_indexes = get_dataset_filelist(opt, mode)

    train_loader = get_loader(training_indexes, opt.downsample_rate, opt.batchsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    ### training epochs for loop ###
    for epoch in range(1, opt.epoch+1):
        train(opt, train_loader, model, optimizer, epoch, validation_indexes)
