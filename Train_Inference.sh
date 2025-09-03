#!/bin/bash
##### 1 #####
python train.py --n_layer 5 --bidirectional True &&
python inference.py --n_layer 5 --bidirectional True &&
python evaluate.py
