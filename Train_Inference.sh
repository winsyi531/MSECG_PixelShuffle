##### 1 #####
python train.py --mamba_in_ch 128 --n_layer 1 --bidirectional False &&
python inference.py --mamba_in_ch 128 --n_layer 1 --bidirectional False &&
python evaluate.py
