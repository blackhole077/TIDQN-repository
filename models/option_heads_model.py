import keras.backend as K
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, Permute, concatenate, Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model

def option_heads_model(input_shape, window_length, nb_actions, num_heads=1, second_input_shape=None):
    option_heads = []
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
    if second_input_shape:
        cond_matrix_input = Input(shape=second_input_shape, name='Conditional_Matrix_Input')
    else:
        cond_matrix_input = None
    ### OPTION HEADS ###
    for i in range(num_heads):
        option_heads.append(option_head(mdp_flatten, nb_actions, i, cond_matrix_input, second_input_shape))
    # Create the overall model (mdp_input -> LazyFrames, cond_matrix_input -> Conditional Matrix)
    if second_input_shape:
        model = Model(inputs=[mdp_input, cond_matrix_input], outputs=option_heads)
    else:
        model = Model(inputs=[mdp_input], outputs=option_heads)
    return model

def option_head(head_input, nb_actions, head_counter, conditional_matrix_input=None, conditional_matrix_input_shape=None):
    mdp_first_dense = Dense(512, activation='relu', name=f'mdp_first_dense_{head_counter}')(head_input)
    if conditional_matrix_input is not None:
        if np.array(conditional_matrix_input_shape).size == 1:
            merged = concatenate([mdp_first_dense, conditional_matrix_input], axis=1)
        elif np.array(conditional_matrix_input_shape).size > 1:
            cond_matrix_flatten = Flatten(name=f'cond_matrix_flatten_{head_counter}')(conditional_matrix_input)
            merged = concatenate([mdp_first_dense, cond_matrix_flatten], axis=1)
        else:
            assert False, 'Wrong second input dimension!'
        output = Dense(nb_actions, activation='linear', name=f'output_{head_counter}')(merged)
    else:
        # Regular Atari head
        output = Dense(nb_actions, activation='linear', name=f'output_{head_counter}')(mdp_first_dense)
    return output
