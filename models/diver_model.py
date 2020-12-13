import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (AveragePooling2D, Conv2D, Cropping2D, Dense, Dropout,
                          Flatten, Input, Lambda, Layer, Permute)
from keras.models import Model

def diver_model(input_shape, diver_weights):
    # MDP Input refers to the game frames (images)
    mdp_shape = (1, ) + input_shape
    image_input = Input(shape=mdp_shape, name='MDP_Diver_Locator_Layer')
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        permutation_layer = Permute((2, 3, 1), input_shape=mdp_shape)
    elif K.image_dim_ordering() == 'channels_first':
        # (channels, width, height)
        permutation_layer = Permute((1, 2, 3), input_shape=mdp_shape)
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    # Reorder the image appropriately
    mdp_permute = permutation_layer(image_input)
    # Crop the image to remove irrelevant details
    cropping_layer = Cropping2D(cropping=(20, 0))(mdp_permute)
    mdp_conv2d_0 = Conv2D(32, (8, 8), activation='relu', strides=(4, 4), name='mdp_conv2d_0')(cropping_layer)
    mdp_conv2d_1 = Conv2D(64, (4, 4), activation='relu', strides=(2, 2), name='mdp_conv2d_1')(mdp_conv2d_0)
    mdp_conv2d_2 = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), name='mdp_conv2d_2')(mdp_conv2d_1)
    # Perform average pooling (use zero padding to keep shape same)
    pooling_layer = AveragePooling2D(padding="same", name="Diver_Locator_AP_1")(mdp_conv2d_2)
    diver_flatten = Flatten(name="Diver_Locator_Flatten")(pooling_layer)
    diver_first_dense = Dense(256, activation='relu', name="Diver_Locator_First_Dense")(diver_flatten)
    diver_first_dropout = Dropout(0.5, name="Diver_Locator_First_Dropout")(diver_first_dense)
    diver_second_dense = Dense(256, activation='relu', name="Diver_Locator_Second_Dense")(diver_first_dropout)
    diver_second_dropout = Dropout(0.5, name="Diver_Locator_Second_Dropout")(diver_second_dense)
    diver_third_dense = Dense(256, activation='relu', name="Diver_Locator_Third_Dense")(diver_second_dropout)
    diver_third_dropout = Dropout(0.5, name="Diver_Locator_Third_Dropout")(diver_third_dense)
    diver_fourth_dense = Dense(256, activation='relu', name="Diver_Locator_Fourth_Dense")(diver_third_dropout)
    diver_fourth_dropout = Dropout(0.5, name="Diver_Locator_Fourth_Dropout")(diver_fourth_dense)
    output = Dense(5, activation='sigmoid', name="Diver_Locator_Output")(diver_fourth_dropout)

    # Create the overall model (mdp_input -> LazyFrames, cond_matrix_input -> Conditional Matrix)
    diver_model = Model(inputs=image_input, outputs=output)
    diver_model.load_weights(diver_weights)
    return diver_model
