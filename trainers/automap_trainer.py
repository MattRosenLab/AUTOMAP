import tqdm
import numpy as np
import tensorflow as tf
import pickle
import sys
import os


class AUTOMAP_Trainer:

    def __init__(self, model, data, valdata, config):

        self.model = model
        self.config = config
        self.data = data
        self.valdata = valdata

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        
    def custom_loss(self,targets,predictions,c_2):

        act_loss = 1e-4*tf.reduce_sum(tf.abs(c_2))

        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.im_h + 8,self.config.im_w + 8])
        predictions = tf.transpose(predictions,perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions,perm=[2,0,1])
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.fc_output_dim])
        
        loss_gradient = tf.reduce_sum(tf.square(targets-predictions)) + act_loss
        
        train_loss_mse_reconstruction = tf.reduce_mean(tf.square(targets-predictions))
        
        return loss_gradient, train_loss_mse_reconstruction

    def valcustom_loss(self,targets,predictions):

        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.im_h + 8,self.config.im_w + 8])
        predictions = tf.transpose(predictions,perm=[1,2,0])
        predictions = tf.image.resize_with_crop_or_pad(predictions, self.config.im_h, self.config.im_w)
        predictions = tf.transpose(predictions,perm=[2,0,1])
        predictions = tf.reshape(predictions,[self.config.batch_size,self.config.fc_output_dim])
        
        val_loss = tf.reduce_mean(tf.square(targets-predictions))
        return val_loss
    
    def train_step(self, epoch, optimizer):

        raw_data, targets = next(self.data.next_batch(self.config.batch_size))
        
        cprob = 1  # multiplicative noise on (default during training)

        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data), minval=0.99,
                                                                      maxval=1.01)) * cprob + raw_data * (1 - cprob)

        with tf.GradientTape() as tape:
            c_2,predictions = self.model(raw_data_input, training=False)
            loss_gradient, train_loss_mse_reconstruction = self.custom_loss(targets,predictions,c_2)
            
        gradients = tape.gradient(loss_gradient, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss = train_loss_mse_reconstruction
        
        return self.train_loss

    def val_step(self, epoch):
        raw_valdata, valtargets = next(self.valdata.next_batch(self.config.batch_size))
        vc_2,predictions = self.model(raw_valdata, training=False)
        valloss = self.valcustom_loss(valtargets,predictions)
        
        self.val_loss = valloss
        return self.val_loss


    def train(self):

        loss_training = np.zeros((2,self.config.num_epochs))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):

            pbar = tqdm.tqdm(total=self.data.len // self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)

            for step in range(self.data.len // self.config.batch_size):
                loss = self.train_step(epoch, optimizer)
                valloss = self.val_step(epoch)
                
                train_status.set_description_str(f'Epoch: {epoch} Loss: {self.train_loss} Val Loss: {self.val_loss}')
                
                pbar.update()

            template = 'Epoch {}, Loss: {}, ValLoss: {}'
            print(template.format(epoch, self.train_loss, self.val_loss))
            
            loss_training[0, epoch] = self.train_loss
            loss_training[1, epoch] = self.val_loss
        self.model.save(self.config.checkpoint_dir)
            # To save a different model/checkpoint at each epoch (will take up a lot more disk space!):
            # self.model.save(os.path.join(self.config.checkpoint_dir,str(epoch)+'.h5'))

        with open(self.config.graph_file, 'wb') as f:
            np.save(f, loss_training)