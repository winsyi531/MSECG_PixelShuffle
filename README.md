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
      │     ├── generate_index.py               # Generate indexes for three datasets
      │     └── loss_function.py                # Every loss function is here
      │
      ├── train.py                              # Training process
      ├── inference.py                          # Inference process, results are stored in sr_signal
      ├── evaluate.py                           # Evaluate the performance between results and GT
      └── Train_Inference.sh                    # Scripting file through all process

## Usage
> It is recommended running the codes in virtual environments, such as [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).

### ※ Training process:
- If you don't want to set training with arguments, you can modify the "default" section in train.py, and simply use
```
python train.py
```

- If regenerating the dataset index is necessary, you can specify the argument by
```
python train.py --regenerate_index True
```

- You should also modify the path to your dataset by
```
python train.py --dataset_dir '/path/to/dataset/'
```

- Hyperparameters, such as batch size, epoch, and down-sampling rate, can also be specified by
```
python train.py --batchsize 16 --epoch 100 --downsample_rate 10
```

- Training loss functions (mse, mse_mag_compm mse_com_comp) and compress factor for the spectrogram can be specified by
```
python train.py --loss mse_mag --compress_factor 0.3
```

### ※ Inference process:
- SR signals will be stored in the 'sr_signal' folder
```
python inference.py
```

### ※ Evaluation process:
- Evaluate the performance by five evaluation metrics, you can use
```
python evaluate.py
```

## Performance

 Model | Train Loss | MSE $\downarrow$ | RMSE $\downarrow$ | SSIM $\uparrow$ | SNR $\uparrow$ | PSNR $\uparrow$ |
  ---  |    ---     |    ---           |     ---           |         ---     |         ---    |         ---     |
 SRECG |    MSE     |     0.000421     |      0.018314     |     0.987028    |    21.484913   |    40.013966    |
 &nbsp;|  MSE+Mag.  |     0.024077     |      0.133012     |     0.786101    |     4.416252   |    23.329115    |

> $x_i \rightarrow$ predicted SR signal

> $y_i \rightarrow$ ground truth SR signal

> $MSE(x_{i},y_{i})=\dfrac{1}{N} \Sigma_{i=1}^{N} (x_i-y_i)^{2}$

> $RMSE(x_{i},y_{i})=\sqrt{MSE(x_{i},y_{i})}$

> $SSIM(x,y)=\dfrac{(2\mu_{x}\mu_{y}+C_{1})(2\sigma_{xy}+C_{2})}{(\mu_{x}^{2}+\mu_{y}^{2}+C_{1})(\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2})}$

> $SNR(x_{i},y_{i})=10\cdot \log_{10}\left(\dfrac{\Sigma_{i=1}^{N}x_{i}^{2}}{\Sigma_{i=1}^{N}(x_{i}-y_{i})^{2}}\right)$

> $PSNR(x_{i},y_{i})=10\cdot \log_{10}\left( \dfrac{\left[\max(x_i)\right]^{2}}{MSE(x_{i},y_{i})}\right)$

## BibTeX
If you want to use this code, please cite the paper below.
```bibtex
@article{chen2023srecg,
  title={SRECG: ECG Signal Super-Resolution Framework for Portable/Wearable Devices in Cardiac Arrhythmias Classification},
  author={Chen, Tsai-Min and Tsai, Yuan-Hong and Tseng, Huan-Hsin and Liu, Kai-Chun and Chen, Jhih-Yu and Huang, Chih-Han and Li, Guo-Yuan and Shen, Chun-Yen and Tsao, Yu},
  journal={IEEE Transactions on Consumer Electronics},
  volume={69},
  number={3},
  pages={250--260},
  year={2023},
  publisher={IEEE}
}
```

