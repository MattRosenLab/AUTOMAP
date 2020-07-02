import numpy as np
import tensorflow as tf
import mat73

import sys
import os

class DataGenerator:
    def __init__(self, config):
        self.config = config

        train_in_file = os.path.join(self.config.data_dir,self.config.train_input)
        train_out_file = os.path.join(self.config.data_dir,self.config.train_output)

        print('*** LOADING TRAINING INPUT DATA ***')
        train_in_dict = mat73.loadmat(train_in_file)
        
        print('*** LOADING TRAINING OUTPUT DATA ***')
        train_out_dict = mat73.loadmat(train_out_file)

        train_in_key = list(train_in_dict.keys())[0]
        train_out_key = list(train_out_dict.keys())[0]

        self.input = np.transpose(train_in_dict[train_in_key])
        self.output = np.transpose(train_out_dict[train_out_key])

        self.len = self.input.shape[0]

    def next_batch(self, batch_size):
        idx = np.random.choice(self.len, batch_size)
        yield self.input[idx], self.output[idx]
