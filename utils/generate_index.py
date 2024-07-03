import os
import random
import numpy as np
import pandas as pd

### Read csv file and write the corresponding train / validation / testing file names to txt files
def generate(opt, train_fold, val_fold, test_fold):
    csv = pd.read_csv(opt.dataset_dir+'ptbxl_database.csv', index_col='ecg_id')
    fold = np.array([f for f in csv.strat_fold])
    file_name = [opt.dataset_dir+fn for fn in csv.filename_hr]

    train_file = [sig for i, sig in enumerate(file_name) if fold[i] in train_fold]
    val_file = [sig for i, sig in enumerate(file_name) if fold[i] in val_fold]
    test_file = [sig for i, sig in enumerate(file_name) if fold[i] in test_fold]
    
    with open(opt.input_training_file, 'w', encoding='utf-8') as f:
        for i in range(len(train_file)):
            f.write(train_file[i]+'\n')
    with open(opt.input_validation_file, 'w', encoding='utf-8') as f:
        for i in range(len(val_file)):
            f.write(val_file[i]+'\n')
    with open(opt.input_testing_file, 'w', encoding='utf-8') as f:
        for i in range(len(test_file)):
            f.write(test_file[i]+'\n')