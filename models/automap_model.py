import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *

def AUTOMAP_Basic_Model(config):

    fc_1 = keras.Input(shape=(config.fc_input_dim), name='input')
    with tf.device('/gpu:0'):
        fc_2 = layers.Dense(config.fc_hidden_dim, activation='tanh')(fc_1)
    with tf.device('/gpu:1'):
        fc_3 = layers.Dense(config.fc_output_dim, activation='tanh')(fc_2)

    fc_3 = layers.Reshape((config.im_h,config.im_w,1))(fc_3)
    
    fc_3 = layers.ZeroPadding2D(4)(fc_3)
    
    c_1 = layers.Conv2D(64,5,strides=1,padding='same',activation='relu')(fc_3)
    c_2 = layers.Conv2D(64,5,strides=1,padding='same',activation='relu')(c_1)
    
    c_3 = layers.Conv2DTranspose(1,7,strides=1,padding='same')(c_2)
    
    output = layers.Reshape(((config.im_h + 8)*(config.im_w + 8),))(c_3) 

    model = keras.Model(inputs = fc_1,outputs = [c_2,output], name='output')
    model.summary()
    return model
