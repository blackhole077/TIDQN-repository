import itertools
import os
import sys

import cv2
import keras.backend as K
import keras.losses
import keras.preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from keras.layers import Activation, Conv2D, Dense, Flatten, Input, Permute
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
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
            image_paths.append(image_path)
            image_list.append(img)
    images = np.array(image_list)
    images = np.reshape(images, (-1, 4, 84, 84))
    labels = np.array(label_list)
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)))
    print("Shape of images: {}".format(images.shape))
    print("Shape of labels: {}".format(labels.shape))
    return images, labels, len(directories), class_weights

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
    base_model.load_weights(base_weights)
    diver_model = Sequential()
    for layer in base_model.layers[:-1]:
        layer.trainable = False
        diver_model.add(layer)
    diver_model.layers[-1].trainable = True
    diver_model.add(Dense(32, activation='relu'))
    diver_model.add(Dropout(0.25))
    diver_model.add(Dense(32, activation='relu'))
    diver_model.add(Dropout(0.25))
    diver_model.add(Dense(num_classes, activation='softmax'))
    diver_model = keras.models.model_from_json(diver_model.to_json())
    diver_model.summary()
    return diver_model

image_paths = []
base_name = "C:\\Users\\Jeevan Rajagopal\\master-thesis-repository\\frames"
images, labels, num_classes, class_weights = build_dataset(base_name)
print("Dim images:{}".format(images.shape))
# print(images[0])
print("Null Values:{}".format(np.argwhere(np.isnan(images))))
# print("Null Values:{}".format(np.argwhere(np.isnan(labels))))
# print("Dim images:{}".format(images.shape))
# print(labels[0])
indices = np.arange(images.shape[0])
(trainX, testX, trainY, testY, _, test_indices) = train_test_split(images,
	labels, indices, test_size=0.25, stratify=labels)
print("Class Weights: {}".format(class_weights))

# model = build_diver_finder_model(output_size=num_classes)
base_model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, NB_ACTIONS)
weights_location = "weights/env=SeaquestDeterministic-v4-c=None-arc=original-mode=off-ns=5000000-seed=64251_weights.h5f"
new_model = build_diver_network(weights_location, num_classes)
opt = SGD(learning_rate=0.01, nesterov=True)
new_model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
train_generator = batch_generator(trainX, trainY, train=True)
valid_generator = batch_generator(testX, testY)
EPOCHS = 3000
file_name = 'diver_locator_weights_dropout_{}.h5'.format(EPOCHS)
model_checkpointer = ModelCheckpoint(file_name, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
plateau_lr = ReduceLROnPlateau(min_lr=0.001, verbose=1)
history = new_model.fit_generator(train_generator,
                            steps_per_epoch= (BATCH_SIZE**2)//BATCH_SIZE,
                            epochs=EPOCHS,
                            class_weight=class_weights,
                            validation_data=valid_generator,
                            validation_steps= (BATCH_SIZE**2) // BATCH_SIZE,
                            callbacks=[model_checkpointer, plateau_lr])
print("Highest Accuracy: {}\n".format(max(history.history['accuracy'])))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("Y True")
print(testY)
test_generator = batch_generator(testX, testY, batch_size=len(testY))
Y_pred = new_model.predict_generator(test_generator, 1)
y_pred = np.argmax(Y_pred, axis=1)
incorrects = np.nonzero(y_pred != testY)[0]
print("Y Predictions")
print(y_pred)
print(incorrects)
label_dict = {
    0: "no_divers",
    1: "up_left",
    2: "down_left",
    3: "up_right",
    4: "down_right"
}
for index in incorrects:
    print("Index: {} Prediction {} True {}".format(test_indices[index], label_dict[y_pred[index]], label_dict[testY[index]]))
    print(image_paths[int(test_indices[index])])

cm = confusion_matrix(testY, y_pred, labels=[0,1,2,3,4])
plot_confusion_matrix(cm=cm, classes=["no_divers", "up_left", "down_left", "up_right", "down_right"], normalize=True)
plt.show()
# print('Classification Report')
# target_names = list(labels)
# print(classification_report(valid_generator.classes, y_pred, target_names=target_names))