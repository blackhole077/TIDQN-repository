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
    ### OPTION HEADS ###
    for _ in range(num_heads):
        option_heads.append(option_head(mdp_flatten, nb_actions, second_input_shape))
    # Creat the overall model (mdp_input -> LazyFrames, cond_matrix_input -> Conditional Matrix)
    if second_input_shape:
        model = Model(inputs=[mdp_input, cond_matrix_input], outputs=option_heads)
    else:
        model = Model(inputs=[mdp_input], outputs=option_heads)
    model.summary()
    return model

def option_head(head_input, nb_actions, conditional_matrix_input_shape=None):
    mdp_first_dense = Dense(512, activation='relu', name='mdp_first_dense')(head_input)
    if conditional_matrix_input_shape:
        conditional_matrix_input = Input(shape=conditional_matrix_input_shape, name='Conditional_Matrix_Input')
        if np.array(conditional_matrix_input_shape).size == 1:
            merged = concatenate([mdp_first_dense, conditional_matrix_input], axis=1)
        elif np.array(conditional_matrix_input_shape).size > 1:
            cond_matrix_flatten = Flatten(name='cond_matrix_flatten')(conditional_matrix_input)
            merged = concatenate([mdp_first_dense, cond_matrix_flatten], axis=1)
        else:
            assert False, 'Wrong second input dimension!'
        output = Dense(nb_actions, activation='linear')(merged)
    else:
        # Regular Atari head
        output = Dense(nb_actions, activation='linear')(mdp_first_dense)
    return output

if __name__ == "__main__":
    nb_actions = 18
    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4
    number_conditionals = 7
    nb_actions = 18
    num_heads = 1
    cond_input_shape = (WINDOW_LENGTH, number_conditionals)
    model = option_heads_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, num_heads, cond_input_shape)
    for i,layer in enumerate(model.layers):
        print(i,layer.name,layer.trainable)