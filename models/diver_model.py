import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Permute, concatenate, Dropout
from keras.models import Model

def diver_model(input_shape, window_length, diver_weights):
    diver_input_shape = (window_length, ) + input_shape
    diver_input = Input(shape=diver_input_shape, name='MDP_Diver_Locator_Layer')
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        diver_permute = Permute((2, 3, 1), input_shape=diver_input_shape)
    elif K.image_dim_ordering() == 'channels_first':
        # (channels, width, height)
        diver_permute = Permute((1, 2, 3), input_shape=diver_input_shape)
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    diver_permute = diver_permute(diver_input)
    diver_first_conv2d = Conv2D(
        32, (8, 8), activation='relu', strides=(4, 4), name='diver_first_conv2d', trainable=False)(diver_permute)
    diver_second_conv2d = Conv2D(
        64, (4, 4), activation='relu', strides=(2, 2), name='diver_second_conv2d', trainable=False)(diver_first_conv2d)
    diver_third_conv2d = Conv2D(
        64, (3, 3), activation='relu', strides=(1, 1), name='diver_third_conv2d', trainable=False)(diver_second_conv2d)
    diver_flatten = Flatten(name='diver_first_flatten', trainable=False)(diver_third_conv2d)
    diver_first_dense = Dense(512, activation='relu', name='diver_first_dense', trainable=False)(diver_flatten)
    diver_first_dropout = Dropout(0.25, name='diver_first_dropout', trainable=False)(diver_first_dense)
    diver_second_dense = Dense(32, activation='relu', name='diver_second_dense', trainable=False)(diver_first_dropout)
    diver_second_dropout = Dropout(0.25, name='diver_second_dropout', trainable=False)(diver_second_dense)
    diver_third_dense = Dense(32, activation='relu', name='diver_third_dense', trainable=False)(diver_second_dropout)
    diver_third_dropout = Dropout(0.25, name='diver_third_dropout', trainable=False)(diver_third_dense)
    # This output is what we want for reward shaping
    diver_output = Dense(5, activation='softmax', name='diver_output', trainable=False)(diver_third_dropout)
    
    diver_model = Model(inputs=diver_input, outputs=diver_output)
    diver_model.load_weights(diver_weights)
    print(diver_model.summary())
    return diver_model
