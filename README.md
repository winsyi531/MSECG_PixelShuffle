# SRECG_torch

This is a pytorch version code for [SRECG: ECG Signal Super-resolution Framework for Portable/Wearable Devices in Cardiac Arrhythmias Classification](https://arxiv.org/abs/2012.03803).

## Introduction

This neural network is designed for Super-Resolution (SR) task, and the architecture is based on [SRResNet](https://arxiv.org/abs/1609.04802).

For now, we use [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset for the experiment.

For evaluation, we follow the metrics (MSE, RMSE, SSIM, SNR, and PSNR) in this paper [DCAE-SR: Design of a Denoising Convolutional Autoencoder for reconstructing Electrocardiograms signals at Super Resolution](https://arxiv.org/abs/2404.15307).

## Data layout

    SRECG
      ├── dataset_index
      │     ├── train.txt                       # Index for training dataset
      │     ├── val.txt                         # Index for validation dataset
      │     └── test.txt                        # Index for testing dataset
      │
      ├── model
      │     ├── model.py                        # Main architecture of SRECG
      │     └── parts.py                        # Modules adopted in the SRECG
      │
      ├── model_pth
      │     └── SRECG-best.pth                  # Checkpoint file for the best performance
      │
      ├── sr_signal                             # Folder for SR signals
      │     ├── SR signal 1
      │     ├──     :
      │     └── SR signal N
      │
      ├── utils                                 # Store functional python files
      │     ├── dataloader.py                   # Methods how to load different datasets
      │     └── generate_index.py               # Generate indexes for three datasets
      │
      ├── train.py                              # Training process
      ├── inference.py                          # Inference process, results are stored in sr_signal
      ├── evaluate.py                           # Evaluate the performance between results and GT
      └── Train_Inference.sh                    # Scripting file through all process

## Usage

### ※ Training process:
If you don't want to set training with arguments, you can modify the "default" section in train.py, and simply use
```
python train.py
```

If regenerating the dataset index is necessary, you can specify the argument by
```
python train.py --regenerate_index True
```

You should also modify the path to your dataset by
```
python train.py --dataset_dir '/path/to/dataset/'
```

Hyperparameters, such as batch size, epoch, and down-sampling rate can also be specified by
```
python train.py --batchsize 16 --epoch 100 --downsample_rate 10
```

### ※ Inference process:
SR signals will be stored in the 'sr_signal' folder
```
python inference.py
```

### ※ Evaluation process:
Evaluate the performance by five evaluation metrics, you can use
```
python evaluate.py
```

