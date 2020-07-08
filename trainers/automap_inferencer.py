import tqdm
import numpy as np
import tensorflow as tf

import sys

class AUTOMAP_Inferencer:

    def __init__(self, model, data, config):

        self.model = model
        self.config = config
        self.data = data

    def inference_step(self,ind_start,batch_size):
        raw_data_input, output = next(self.data.next_batch(ind_start,batch_size))
        predictions = self.model(raw_data_input, training=False)
        return predictions

    def inference(self):
        output_array = [] 
        for step in range(int(np.ceil(self.data.len/self.config.batch_size))):

            if step < np.ceil(self.data.len/self.config.batch_size)-1:
                batch_size = self.config.batch_size
            else:
                batch_size = self.data.len-self.config.batch_size*step
            
            ind_start = step*self.config.batch_size
            predictions = self.inference_step(ind_start,batch_size)

            print(ind_start)
            
        template = 'Epoch {}, Loss: {}'
        print('done')
        

