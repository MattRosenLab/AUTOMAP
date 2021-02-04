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

    def train_step(self, epoch, loss_object, optimizer):

        raw_data, targets = next(self.data.next_batch(self.config.batch_size))
        
        cprob = 1  # multiplicative noise on (default during training)

        raw_data_input = tf.math.multiply(raw_data, tf.random.uniform(shape=tf.shape(raw_data), minval=0.99,
                                                                      maxval=1.01)) * cprob + raw_data * (1 - cprob)

        with tf.GradientTape() as tape:
            predictions = self.model(raw_data_input, training=False)
            loss = loss_object(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def val_step(self, epoch, valloss_object):
        raw_valdata, valtargets = next(self.valdata.next_batch(self.config.batch_size))
        predictions = self.model(raw_valdata, training=False)
        valloss = valloss_object(valtargets, predictions)
        self.val_loss(valloss)

    def train(self):

        loss_training = np.zeros((2,self.config.num_epochs))
        #valloss_training = np.zeros(self.config.num_epochs)
        loss_object = tf.keras.losses.MeanSquaredError()
        valloss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            self.train_loss.reset_states()
            self.val_loss.reset_states()

            pbar = tqdm.tqdm(total=self.data.len // self.config.batch_size, desc='Steps', position=0)
            train_status = tqdm.tqdm(total=0, bar_format='{desc}', position=1)

            for step in range(self.data.len // self.config.batch_size):
                loss = self.train_step(epoch, loss_object, optimizer)
                valloss = self.val_step(epoch, valloss_object)
                train_status.set_description_str(f'Epoch: {epoch} Loss: {self.train_loss.result()} Val Loss: {self.val_loss.result()}')
                pbar.update()

            template = 'Epoch {}, Loss: {}, ValLoss: {}'
            print(template.format(epoch, self.train_loss.result(), self.val_loss.result()))
            
            loss_training[0, epoch] = self.train_loss.result()
            loss_training[1, epoch] = self.val_loss.result()
        self.model.save(self.config.checkpoint_dir)
            # To save a different model/checkpoint at each epoch (will take up a lot more disk space!):
            # self.model.save(os.path.join(self.config.checkpoint_dir,str(epoch)+'.h5'))

        with open(self.config.graph_file, 'wb') as f:
            np.save(f, loss_training)