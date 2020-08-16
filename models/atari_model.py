from keras.layers import Activation, Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
import keras.backend as K


def atari_model(input_shape, window_length, nb_actions):
    input_shape = (window_length, ) + input_shape
    model = Sequential()
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'channels_first':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    # model.add(Activation('linear'))
    model.summary()
    return model