### import library ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from model.model import SRECG
from utils.dataloader import get_loader, test_dataset, get_dataset_filelist
from utils.generate_index import generate
from utils.loss_function import MSE_LOSS, MAG_MSE_Loss, COM_MSE_Loss, RI_MSE_Loss
from utils.utils import adjust_lr
import numpy as np
from tqdm import tqdm
from torchinfo import summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
############################################
log_loss = []
### construct loss function for training ###
def structure_loss(opt, pred, gt):
    """
    input:
        pred: output from network
        mask: ground truth
    output:
        loss value (mse+STFT_MSE)
    """
    # Pure MSE Loss
    if opt.loss == 'mse':
        loss_mse = MSE_LOSS(pred, gt)
        score = loss_mse
    # MSE Loss + compressed Magnitude Loss
    elif opt.loss == 'mse_mag':
        loss_mse = MSE_LOSS(pred, gt)
        loss_mag = MAG_MSE_Loss(pred, gt, opt.n_fft, opt.compress_factor)
        score = loss_mse + (0.1 * loss_mag)
    # MSE Loss + compressed Complex Loss
    elif opt.loss == 'mse_com':
        loss_mse = MSE_LOSS(pred, gt)
        loss_com = COM_MSE_Loss(pred, gt, opt.n_fft, opt.compress_factor)
        score = loss_mse + loss_com
    elif opt.loss == 'mse_ri':
        loss_mse = MSE_LOSS(pred, gt)
        loss_ri = RI_MSE_Loss(pred, gt, opt.n_fft, opt.compress_factor)
        score = loss_mse + (0.1 * loss_ri)
    
    return score

### validation process during training ###
def val(opt, model, val_index, epoch):
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
    test_loader = test_dataset(val_index, opt)
    num1 = len(val_index)
    MSE = 0.0
    for i in tqdm(range(num1), (f'Epoch[{epoch}/{opt.epoch}]'), unit=' signal'):
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

### training process ###
def train(opt, train_loader, model, optimizer, epoch, decay_epoch, validation_indexes):
    if epoch == decay_epoch:
        model.load_state_dict(torch.load(opt.train_save+'SRECG-best.pth'))
        print('Adopting the best checkpoint for following decayed learning rate.')
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
        loss = structure_loss(opt, sr_audio, gt_audio)
        # ---- backward ----
        loss.backward()
        optimizer.step()
        # ---- tqdm update ---- #
        loop.set_description(f'Epoch[{epoch}/{opt.epoch}]')
        loop.set_postfix(loss=loss.item())
        log_loss.append(loss.item())

    # output an image recording iteration loss
    plt.figure()
    plt.plot(log_loss, label='Ite. Loss')
    plt.title('Iteration Loss', fontsize=20)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('WAV(MSE)', fontsize=12)
    plt.legend()
    plt.grid(which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('./iteration_loss.png', transparent=False, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # specify the validation interval
    # default was one epoch one validation
    if epoch % 1 == 0:
        dataset = 'PTB-XL'
        valid_loss = val(opt, model, validation_indexes, epoch) # use above val() function
        print(dataset, ':', valid_loss)

        if (epoch==1) or (valid_loss<best):
            best = valid_loss # store best Dice Score
            torch.save(model.state_dict(), save_path+'SRECG-best.pth') # save best model
            print('########## Best Model Saved, MSE: {:.8f} ##########'.format(best))
####################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--epoch', type=int,
                        default=350, help='epoch number')
    parser.add_argument('--downsample_rate', type=int,
                        default=10, help='how many times to down sample the signals')
    parser.add_argument('--compress_factor', type=float,
                        default=1.0, help='factor compressing the magnitude')
    parser.add_argument('--n_fft', type=int,
                        default=64, help='number of points for converting waveform into frequency domain')
    parser.add_argument('--mamba_in_ch', type=int,
                        default=128, help='channels for input of Mamba block')
    parser.add_argument('--n_layer', type=int,
                        default=1, help='number of layers for Mamba block')
    parser.add_argument('--bidirectional', type=str,
                        default='False', help='True or False for Bi-Directional Mamba')
    parser.add_argument('--loss', type=str,
                        default='mse', help='factor compressing the magnitude')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=300, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--train_save', type=str,
                        default='./model_pth/')
    parser.add_argument('--regenerate_index', default=False, help='whether want to generate the dataset')
    parser.add_argument('--dataset_dir', type=str, default='/dataset/PTB-XL/')
    parser.add_argument('--noise_dir', type=str, default='/dataset/MIT-BIH_noise/')
    parser.add_argument('--input_training_file', type=str, default='./dataset_index/train.txt')
    parser.add_argument('--input_validation_file', type=str, default='./dataset_index/val.txt')
    parser.add_argument('--input_testing_file', type=str, default='./dataset_index/test.txt')
    opt = parser.parse_args()

    # ---- build models ----
    #torch.cuda.set_device(0)  # set your gpu device
    model = SRECG(mamba_in_ch=opt.mamba_in_ch, n_layer=opt.n_layer, bidirectional=opt.bidirectional).cuda()
    
    # ---- summary ---- #
    summary(model, (1, 12, 500)) # show model architecture and parameters
    
    best = 0
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)

    if opt.regenerate_index:
        generate(opt)
    mode = 'train'
    training_indexes, validation_indexes = get_dataset_filelist(opt, mode)

    train_loader = get_loader(training_indexes, opt)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    ### training epochs for loop ###
    for epoch in range(1, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.decay_epoch)
        train(opt, train_loader, model, optimizer, epoch, opt.decay_epoch, validation_indexes)

    
    
