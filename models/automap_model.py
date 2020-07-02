import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def AUTOMAP_Basic_Model(config):

    fc_1 = keras.Input(shape=(config.fc_input_dim), name='input')
    with tf.device('/gpu:0'):
        fc_2 = layers.Dense(config.fc_hidden_dim, activation='tanh')(fc_1)
    with tf.device('/gpu:1'):
        fc_3 = layers.Dense(config.fc_output_dim, activation='tanh')(fc_2)

    fc_3 = layers.Reshape((config.im_h,config.im_w,1))(fc_3)
    
    c_1 = layers.Conv2D(64,5,strides=1,padding='same',activation='relu')(fc_3)
    c_2 = layers.Conv2D(64,5,strides=1,padding='same',activation='relu',activity_regularizer=regularizers.l1(.0001))(c_1)
    c_3 = layers.Conv2DTranspose(1,7,strides=1,padding='same')(c_2)

    output = layers.Reshape((config.fc_output_dim,))(c_3) 

    model = keras.Model(inputs = fc_1,outputs = output, name='output')
    model.summary()
    return model

