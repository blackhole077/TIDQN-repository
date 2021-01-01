import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Permute, concatenate, Dropout
from keras.models import Model

def merged_model(input_shape, window_length, nb_actions, second_input_shape):
    # MDP Input refers to the game frames (images)
    mdp_shape = (window_length, ) + input_shape
    mdp_input = Input(shape=mdp_shape, name='MDP_Diver_Locator_Layer')
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        mdp_permute = Permute((2, 3, 1), input_shape=mdp_shape)
    elif K.image_dim_ordering() == 'channels_first':
        # (channels, width, height)
        mdp_permute = Permute((1, 2, 3), input_shape=mdp_shape)
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    mdp_permute = mdp_permute(mdp_input)
    mdp_first_conv2d = Conv2D(
        32, (8, 8), activation='relu', strides=(4, 4), name='mdp_first_conv2d')(mdp_permute)
    mdp_second_conv2d = Conv2D(
        64, (4, 4), activation='relu', strides=(2, 2), name='mdp_second_conv2d')(mdp_first_conv2d)
    mdp_third_conv2d = Conv2D(
        64, (3, 3), activation='relu', strides=(1, 1), name='mdp_third_conv2d')(mdp_second_conv2d)
    mdp_flatten = Flatten(name='mdp_first_flatten')(mdp_third_conv2d)
    mdp_first_dense = Dense(512, activation='relu', name='mdp_first_dense')(mdp_flatten)

    cond_matrix_input = Input(shape=second_input_shape, name='Conditional_Matrix_Input')
    if np.array(second_input_shape).size == 1:
        merged = concatenate([mdp_first_dense, cond_matrix_input], axis=1)
    elif np.array(second_input_shape).size > 1:
        cond_matrix_flatten = Flatten(name='cond_matrix_flatten')(cond_matrix_input)
        merged = concatenate([mdp_first_dense, cond_matrix_flatten], axis=1)
    else:
        assert False, 'Wrong second input dimension!'

    output = Dense(nb_actions, activation='linear')(merged)
    # Creat the overall model (mdp_input -> LazyFrames, cond_matrix_input -> Conditional Matrix)
    model = Model(inputs=[mdp_input, cond_matrix_input], outputs=output)
    model.summary()
    return model

# INPUT_SHAPE = (84, 84)
# WINDOW_LENGTH = 4
# number_conditionals = 3
# nb_actions = 18
# cond_input_shape = (WINDOW_LENGTH, number_conditionals)
# model = merged_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, cond_input_shape)
# for i,layer in enumerate(model.layers):
#     print(i,layer.name,layer.trainable)