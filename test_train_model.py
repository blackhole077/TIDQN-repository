import os
import sys

import cv2
import keras.backend as K
import keras.losses
import keras.preprocessing.image
import numpy as np
import tensorflow as tf
# from keras.layers import Activation, Conv2D, Dense, Flatten, Input, Permute
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import models.atari_model as atari_model

WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)
NB_ACTIONS = 18
BATCH_SIZE= 256

def build_dataset(base_path=None):
    image_list = []
    label_list = []
    label_map_dict = {
        "no_divers": 0,
        "up_left": 1,
        "down_left": 2,
        "up_right": 3,
        "down_right": 4
    }
    # label_map_dict={
    #     "up_left": 0,
    #     "down_left": 1,
    #     "up_right": 2,
    #     "down_right": 3
    # }
    directories = [_dir for _dir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, _dir))]
    print("Directories: {}".format(directories))
    for class_name in directories:
        class_path = os.path.join(base_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = Image.open(image_path)
            img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
            # img.show()
            img = np.array(img).astype(np.float32)
            img = img / 255.
            # Make Lazy Frame stack of 4 frames
            img = np.repeat(img[:, :, np.newaxis], 4, axis=2)
            label_to_append = label_map_dict[class_name]
            if label_to_append is None:
                raise ValueError("Expected a label for {} got None.".format(class_name))
            label_list.append(label_map_dict.get(class_name))

            image_list.append(img)
    images = np.array(image_list)
    images = np.reshape(images, (-1, 4, 84, 84))
    labels = np.array(label_list)
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(labels), labels)))
    print("Shape of images: {}".format(images.shape))
    print("Shape of labels: {}".format(labels.shape))
    return images, labels, len(directories), class_weights

def batch_generator(X, Y, batch_size = BATCH_SIZE):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]


base_name = "C:\\Users\\Jeevan Rajagopal\\master-thesis-repository\\frames"
images, labels, num_classes, class_weights = build_dataset(base_name)
print("Dim images:{}".format(images.shape))
# print(images[0])
print("Null Values:{}".format(np.argwhere(np.isnan(images))))
# print("Null Values:{}".format(np.argwhere(np.isnan(labels))))
# print("Dim images:{}".format(images.shape))
# print(labels[0])
(trainX, testX, trainY, testY) = train_test_split(images,
	labels, test_size=0.25)
print("Class Weights: {}".format(class_weights))

# model = build_diver_finder_model(output_size=num_classes)
base_model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, NB_ACTIONS)
weights_location = "weights/env=SeaquestDeterministic-v4-c=None-arc=original-mode=off-ns=5000000-seed=64251_weights.h5f"
base_model.load_weights(weights_location)
base_model.summary()
# model._layers.pop(1)
# model._layers[-1] = Dense(5, activation='linear')(model._layers[-2])
# print(model._layers[1].batch_input_shape)
# model.summary()
print(base_model._layers[2])
new_model = Sequential()
# input_section.add(Input(shape=(84,84,1)))
# new_model.add(Conv2D(32, (8, 8), activation='relu', strides=(4, 4), input_shape=(84,84,1)))
# print(base_model._layers)
# print(base_model._layers[2].batch_input_shape)
# new_model._layers[0].set_weights([base_model.layers[2].get_weights()])
# new_model = Sequential()
# new_model.add(new_input)
for layer in base_model.layers[:-1]:
    layer.trainable = False
    new_model.add(layer)
# Only set the Dense Layer to be trainable.
new_model.layers[-1].trainable = True
# Change the output to match new labels
new_model.add(Dense(32, activation='relu'))
new_model.add(Dropout(0.25))
new_model.add(Dense(32, activation='relu'))
new_model.add(Dropout(0.25))
new_model.add(Dense(5, activation='softmax'))

new_model = keras.models.model_from_json(new_model.to_json())
new_model.compile(optimizer='sgd', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# new_model.summary()
train_generator = batch_generator(trainX, trainY)
valid_generator = batch_generator(testX, testY)
# h = new_model.fit(x=trainX, y=trainY, batch_size=128, epochs=10, validation_data=(testX, testY))

model_checkpointer = ModelCheckpoint('diver_locator_weights_dropout.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)

history = new_model.fit_generator(train_generator,
                            steps_per_epoch=65536//BATCH_SIZE,
                            epochs=400,
                            class_weight=class_weights,
                            validation_data=valid_generator,
                            validation_steps=65536 // BATCH_SIZE,
                            callbacks=[model_checkpointer])
print("Highest Accuracy: {}\n".format(max(history.history['accuracy'])))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Y_pred = new_model.predict_generator(valid_generator, 65536 // BATCH_SIZE)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(valid_generator.classes, y_pred))
# print('Classification Report')
# target_names = list(labels)
# print(classification_report(valid_generator.classes, y_pred, target_names=target_names))