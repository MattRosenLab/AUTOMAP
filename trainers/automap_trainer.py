from base.base_train import BaseTrain
import tqdm
import numpy as np
import tensorflow as tf

import sys

class AUTOMAP_Trainer:


    def __init__(self, model, data, config):

        self.model = model
        self.config = config
        self.data = data
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def train_step(self,epoch,loss_object,optimizer):

        raw_data, targets = next(self.data.next_batch(self.config.batch_size))

        if self.config.train_flag == 1: 
            cprob = 1
        else:
            cprob = 0

        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data),minval=0.99, maxval=1.01)) * cprob + raw_data * (1 - cprob)

        with tf.GradientTape() as tape:
            predictions = self.model(raw_data_input, training=False)
            loss = loss_object(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)



    def train(self):
        
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = config.learning_rate)

        for epoch in range(self.config.num_epochs):
            self.train_loss.reset_states()

            pbar = tqdm.tqdm(total=self.data.len//self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)


            for step in range(self.data.len//self.config.batch_size):
                loss = self.train_step(epoch,loss_object, optimizer)
                train_status.set_description_str(f'Epoch: {epoch} Loss: {self.train_loss.result()}')
                pbar.update()

            template = 'Epoch {}, Loss: {}'
            print(template.format(epoch, self.train_loss.result()))

            self.model.save(self.config.checkpoint_dir)

            # To save a different model/checkpoint at each epoch (will take up a lot more disk space!):
            # self.model.save(os.path.join(self.config.checkpoint_dir,str(epoch)))    
                   

        

