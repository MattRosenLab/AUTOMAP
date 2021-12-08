import tqdm
import numpy as np
import tensorflow as tf
import scipy.io as sio
import sys

class AUTOMAP_Inferencer:

    def __init__(self, model, data, config):

        self.model = model
        self.config = config
        self.data = data

    def inference_step(self,ind_start,batch_size):
        raw_data_input, output = next(self.data.next_batch(ind_start,batch_size))
        c_2, predictions = self.model(raw_data_input, training=False)
        return predictions

    def inference(self):
        output_array = np.empty((self.data.len,self.config.fc_output_dim)) 
        # output_array = np.array([])
        for step in range(int(np.ceil(self.data.len/self.config.batch_size))):

            if step < np.ceil(self.data.len/self.config.batch_size)-1:
                batch_size = self.config.batch_size
            else:
                batch_size = self.data.len-self.config.batch_size*step
            
            ind_start = step*self.config.batch_size
            predictions = self.inference_step(ind_start,batch_size)
            
            bs = predictions.shape[0]
            predictions = tf.reshape(predictions,[bs,self.config.im_h + 8,self.config.im_h + 8])
            predictions = tf.transpose(predictions,perm=[1,2,0])
            predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_h)
            predictions = tf.transpose(predictions,perm=[2,0,1])
            predictions = tf.reshape(predictions,[bs,self.config.fc_output_dim])            
            
            output_array[ind_start:ind_start+batch_size,:]=predictions

        
        sio.savemat(self.config.save_inference_output,{'output_array':output_array})

        print('Inference Done')
            

