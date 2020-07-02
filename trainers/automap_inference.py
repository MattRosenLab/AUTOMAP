import tqdm
import numpy as np
import tensorflow as tf

import sys

class AUTOMAP_Inferencer:

    def __init__(self, model, data, config):

        self.model = model
        self.config = config
        self.data = data

    def inference_step(self):
        raw_data = next(self.data.next_batch(self.config.batch_size))
        predictions = self.model(raw_data_input, training=False)
        return predictions

    def inference(self):
        output_array = [] 
        for step in range(np.ceil(self.data.len/self.config.batch_size)):
            predictions = self.inference_step()
            
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch, self.train_loss.result()))

        self.model.save(self.config.checkpoint_dir)

            # To save a different model/checkpoint at each epoch (will take up a lot more disk space!):
            # self.model.save(os.path.join(self.config.checkpoint_dir,str(epoch)))    
                   

        

