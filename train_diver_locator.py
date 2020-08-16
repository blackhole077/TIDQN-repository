import os
import sys
import numpy as np
import tensorflow as tf
import keras.losses
from keras.layers import Activation, Conv2D, Dense, Flatten, Permute, Input
from keras.models import Sequential
import keras.backend as K
print(tf.__version__)
from keras.optimizers import Adam
INPUT_SHAPE=(84,84)
WINDOW_LENGTH = 4
batch_size = 32
epochs=100

def read_data(base_path, class_names):
    image_list = []
    label_list = []
    label_map_dict = {}
    count_label = 0

    directories = [_dir for _dir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, _dir))]
    print("Directories: {}".format(directories))
    for class_name in directories:
        class_path = os.path.join(base_path, class_name)
        label_map_dict[class_name]=count_label

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            label_list.append(count_label)
            image_list.append(image_path)

        count_label += 1
    return image_list, label_list, label_map_dict

def _parse_function(filename, label):
    print("File Name: {}".format(filename))
    print("Label: {}".format(label))
    image_string = tf.io.read_file(filename, "file_reader")
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    processed_image = tf.image.resize(image_decoded,INPUT_SHAPE)
    processed_image = tf.cast(processed_image, tf.uint8)
    print(processed_image)
    return processed_image, label

def loss(labels, logits):
    return keras.losses.categorical_crossentropy(labels, logits)

def tfdata_generator(images, labels, batch_size=128, shuffle=True):

    def _parse_function(filename, label):
        # print("File Name: {}".format(filename))
        # print("Label: {}".format(label))
        image_string = tf.io.read_file(filename, "file_reader")
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        processed_image = tf.image.resize(image_decoded,INPUT_SHAPE)
        processed_image = tf.cast(processed_image, tf.uint8)
        return [processed_image, label]

    # def map_func(image, label):
    #     '''A transformation function'''
    #     x_train = tf.reshape(tf.cast(image, tf.float32), image_shape)
    #     y_train = tf.one_hot(tf.cast(label, tf.uint8), num_classes)
    #     return [x_train, y_train]

    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels)))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(len(images)).batch(batch_size).repeat()
    # print("Dataset Shape: {}".format(tf.compat.v1.data.get_output_shapes(dataset)))

    # dataset  = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset  = dataset.map(map_func)
    # dataset  = dataset.shuffle().batch(batch_size).repeat()
    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    # iterator.initializer()
    next_batch = iterator.get_next()
    print(next_batch)
    while True:
        yield K.get_session().run(next_batch)

# tf.compat.v1.enable_eager_execution()

base_path = "C:\\Users\\Jeevan Rajagopal\\master-thesis-repository\\frames"
class_names = ["no_divers","up_right","up_left","down_left","down_right"] #Follow the cartesian quadrants
image_list, label_list, label_map_dict = read_data(base_path, class_names)
# print("Keys: {}".format(label_map_dict.keys()))
# print("Example Values: {}".format(label_map_dict.get('down_right')))
# dataset = tf.compat.v1.data.Dataset.from_tensor_slices((tf.constant(image_list), tf.constant(label_list)))
# images_dataset = tf.compat.v1.data.Datset.from_tensor_slices(tf.constant(image_list))
# labels_dataset = tf.compat.v1.data.Datset.from_tensor_slices(tf.constant(label_list))
# dataset = dataset.shuffle(len(image_list))
# dataset = dataset.repeat(epochs)
# dataset = dataset.map(_parse_function).batch(batch_size).repeat()

# print("Dataset Size: {}".format(len(list(dataset))))
# iterator = dataset.make_one_shot_iterator()


input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE
model = Sequential()
# if K.image_data_format() == 'channels_last':
#     # (width, height, channels)
#     model.add(Permute((2, 3, 1), input_shape=input_shape))
# elif K.image_dim_ordering() == 'channels_first':
#     # (channels, width, height)
#     model.add(Permute((1, 2, 3), input_shape=input_shape))
# else:
#     raise RuntimeError('Unknown image_dim_ordering.')
model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(84,84,1)))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(len(class_names)))
model.add(Activation('linear'))
model.compile(optimizer='sgd', loss=keras.losses.SparseCategoricalCrossentropy())
model.summary()

# model.fit(dataset)

model.fit_generator(generator = tfdata_generator(image_list, label_list),
                    steps_per_epoch=200,
                    epochs=epochs,
                    workers = 0 , # This is important
                    verbose = 1)