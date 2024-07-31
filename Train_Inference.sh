#!/bin/bash
##### 1 #####
python train.py --n_layer 1 --bidirectional True &&
python inference.py --n_layer 1 --bidirectional True &&
python evaluate.py && 
python train.py --n_layer 2 --bidirectional True &&
python inference.py --n_layer 2 --bidirectional True &&
python evaluate.py && 
python train.py --n_layer 3 --bidirectional True &&
python inference.py --n_layer 3 --bidirectional True &&
python evaluate.py && 
python train.py --n_layer 4 --bidirectional True &&
python inference.py --n_layer 4 --bidirectional True &&
python evaluate.py && 
python train.py --n_layer 5 --bidirectional True &&
python inference.py --n_layer 5 --bidirectional True &&
python evaluate.py
