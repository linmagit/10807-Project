#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import os
import pdb

def load_data():
    d_prefix='/home/junjuew/deep_learning/data'
    fns = [
    'char_rnn_data_mat_10k.pkl',
    'char_rnn_label_mat_10k.pkl',       
    'char_rnn_data_mat_1k_val.pkl', 
    'char_rnn_label_mat_1k_val.pkl'
    ]
    data=[]
    for idx, fn in enumerate(fns):
        data.append(np.load(os.path.join(d_prefix, fn)))

    trd=(data[0], data[1])
    ted=(data[2], data[3])
    return trd, ted
    
print('training #: {} testing #: {}'.format(len(trd[0]), len(ted[0])))

def get_batch(data, s_idx, e_idx):
    return (data[0][s_idx:e_idx,:,:], data[1][s_idx:e_idx,:,])

def get_batch_num(data, batch_size):
    return int(data[0].shape[0]/ float(batch_size))
