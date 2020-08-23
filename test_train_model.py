import itertools
import os
import sys

import cv2

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import keras.losses
import keras.preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dropout, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import models.atari_model as atari_model
import time
import argparse


WINDOW_LENGTH = 4
INPUT_SHAPE = (84, 84)
NB_ACTIONS = 18
BATCH_SIZE= 256

### VISUALIZATION-RELATED FUNCTIONS ###

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.3f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def build_performance_report(model_history=None, file_name=None):
    print("Highest Accuracy: {}\n".format(max(model_history.history['accuracy'])))
    print("Highest Validation Accuracy: {}\n".format(max(model_history.history['val_accuracy'])))
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy over {} Epochs'.format(len(model_history.history['accuracy'])))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(file_name)
    plt.show()

def build_confusion_matrix(trained_model=None, testX=None, testY=None):
    test_generator = batch_generator(testX, testY, batch_size=len(testY))
    prediction_probabilities = trained_model.predict_generator(test_generator, 1)
    y_pred = np.argmax(prediction_probabilities, axis=1)
    incorrects = np.nonzero(y_pred != testY)[0]
    print("Y True")
    print(testY)
    print("Y Predictions")
    print(y_pred)
    print("Number Incorrect: {}".format(len(incorrects)))
    for index in incorrects:
        print("Index: {} Prediction {} True {}".format(test_indices[index], label_dict[y_pred[index]], label_dict[testY[index]]))
        print(image_paths[int(test_indices[index])])
    return confusion_matrix(testY, y_pred, labels=[0,1,2,3,4])

### MODEL-RELATED FUNCTIONS ###

def build_dataset(base_path=None, label_map_dictionary=None):
    image_list = []
    label_list = []
    image_paths = []

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
            label_to_append = label_map_dictionary[class_name]
            if label_to_append is None:
                raise ValueError("Expected a label for {} got None.".format(class_name))
            label_list.append(label_map_dictionary.get(class_name))
            image_paths.append(image_path)
            image_list.append(img)
    images = np.array(image_list)
    images = np.reshape(images, (-1, 4, 84, 84))
    labels = np.array(label_list)
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)))
    print("Shape of images: {}".format(images.shape))
    print("Shape of labels: {}".format(labels.shape))
    return images, labels, image_paths, class_weights

def batch_generator(X, Y, batch_size = BATCH_SIZE, train=False):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            if train:
                np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]


def build_diver_network(base_weights=None, num_classes=None):
    base_model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, NB_ACTIONS)
    if base_weights:
        base_model.load_weights(base_weights)
    diver_model = Sequential()
    for layer in base_model.layers[:-1]:
        layer.trainable = True
        diver_model.add(layer)
    print(diver_model.layers[-1])
    # diver_model.layers[-1].trainable = True
    # print(diver_model.layers[-3])
    # diver_model.layers[-3].trainable = True
    diver_model.add(Dropout(0.25))
    diver_model.add(Dense(32, activation='relu'))
    diver_model.add(Dropout(0.25))
    diver_model.add(Dense(32, activation='relu'))
    diver_model.add(Dropout(0.25))
    diver_model.add(Dense(num_classes, activation='softmax'))
    diver_model = keras.models.model_from_json(diver_model.to_json())
    diver_model.summary()
    return diver_model

def build_diver_network_from_binary(binary_weights=None, num_classes=None, all_layers_trainable=False):
    base_model = build_diver_network(None, 2)
    if binary_weights:
        base_model.load_weights(binary_weights)
    diver_model = Sequential()
    for layer in base_model.layers[:-1]:
        layer.trainable = all_layers_trainable
        diver_model.add(layer)
    if not all_layers_trainable:
        # Set all Dense layers from intermediate model to be trainable.
        for layer in diver_model.layers[-6:]:
            layer.trainable = True

    diver_model.add(Dense(num_classes, activation='softmax'))
    diver_model = keras.models.model_from_json(diver_model.to_json())
    diver_model.summary()
    return diver_model

def _main(args):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=config))
    print(K.tensorflow_backend._get_available_gpus())
    classification_type = args['classification_type']
    base_weights = args['base_weights']
    if classification_type == 'binary':
        label_map_dict = {
            "no_divers": 0,
            "diver": 1
        }
        base_name = "C:\\Users\\Jeevan Rajagopal\\master-thesis-repository\\diver_frames\\train"
    else:
        label_map_dict = {
            "no_divers": 0,
            "up_left": 1,
            "down_left": 2,
            "up_right": 3,
            "down_right": 4
        }
        base_name = "C:\\Users\\Jeevan Rajagopal\\master-thesis-repository\\frames"
    confusion_matrix_classes = list(label_map_dict.keys())
    nb_classes = len(confusion_matrix_classes)
    if base_weights == 'atari':
        weights_location = "weights/env=SeaquestDeterministic-v4-c=None-arc=original-mode=off-ns=5000000-seed=64251_weights.h5f"
        diver_model = build_diver_network(base_weights=weights_location, num_classes=nb_classes)
    else:
        weights_location = "diver_intermediate_weights_400_diver_no_diver.h5"
        diver_model = build_diver_network_from_binary(binary_weights=weights_location, num_classes=nb_classes)
    
    ### MISCELLANEOUS INITIALIZATIONS ###
    EPOCHS = args['num_epochs']
    timestr = time.strftime("%Y%m%d-%H%M%S")
    weights_file_name = f'diver_locator_{base_weights}_weights_{classification_type}_class_{EPOCHS}_epochs_{timestr}_weights_file.h5'
    figure_file_name = f'diver_locator_{base_weights}_weights_{classification_type}_class_{EPOCHS}_epochs_{timestr}_figure_file.png'
    confusion_matrix_file_name = f'diver_locator_{base_weights}_weights_{classification_type}_class_{EPOCHS}_epochs_{timestr}_confustion_matrix.png'
    ### OPTIMIZER SECTION ###
    opt = SGD(learning_rate=0.001, nesterov=True)
    ### CALLBACK SECTION ###
    model_checkpointer = ModelCheckpoint(weights_file_name, monitor='accuracy', save_best_only=True, save_weights_only=True, verbose=1)
    stopper = EarlyStopping(monitor='accuracy', patience=50, mode='max', restore_best_weights=True)
    plateau_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', min_lr=0.001, factor=0.5, verbose=1)
    # Compile the model
    diver_model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    ### DATASET CONSTRUCTION SECTION ###
    images, labels, image_paths, class_weights = build_dataset(base_name, label_map_dict)
    print("Dim images:{}".format(images.shape))
    # print(images[0])
    print("Null Values:{}".format(np.argwhere(np.isnan(images))))
    # print("Null Values:{}".format(np.argwhere(np.isnan(labels))))
    # print("Dim images:{}".format(images.shape))
    # print(labels[0])
    indices = np.arange(images.shape[0])
    (trainX, testX, trainY, testY, _, test_indices) = train_test_split(images,
        labels, indices, test_size=0.1, stratify=labels)
    print("Class Weights: {}".format(class_weights))
    train_generator = batch_generator(trainX, trainY, train=True)
    valid_generator = batch_generator(testX, testY)
    
    ### MODEL TRAINING SECTION ###
    history = diver_model.fit_generator(train_generator,
                                steps_per_epoch= (BATCH_SIZE**2)//BATCH_SIZE,
                                epochs=EPOCHS,
                                class_weight=class_weights,
                                validation_data=valid_generator,
                                validation_steps= (BATCH_SIZE**2) // BATCH_SIZE,
                                callbacks=[model_checkpointer, stopper])
    ### EVALUATION SECTION ###
    build_performance_report(history, figure_file_name)
    cm = build_confusion_matrix(new_model, testX, testY)

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    '-b',
    '--baseweights',
    choices=['atari', 'binary'],
    dest='base_weights',
    help='Select what base_weights will be used.\
          Atari (default) indicates that the weights and model\
          will be based on Vanilla DQN. Binary indicates\
          that it will be based on an intermediate training\
          step on a binary classification problem.',
    default='atari',
    type=str
)
PARSER.add_argument(
    '-c',
    '--classtype',
    dest='classification_type',
    choices=['binary', 'multi'],
    help='Select what type of problem the model is being trained on.\
          Binary indicates that the model will be trained on the\
          diver/no diver problem. Multi (default) indicates the model\
          will be trained on the quadrants + no diver multi-class problem.',
    default='multi',
    type=str
)
PARSER.add_argument(
    '-e',
    '--epochs',
    dest='num_epochs',
    help='Select how many epochs the model will train.\
          400 epochs is the default. However, early stopping\
          with a patience value of 50 epochs is still in place.',
    default=400,
    type=int
)
_main(vars(PARSER.parse_args()))