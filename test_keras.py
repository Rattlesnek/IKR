import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt

import sys


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
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

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = (len(ims) // rows) if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')



# PATH TO DATA

train_path = 'data/train'
valid_path = 'data/dev' # TODO valid_path is for now the same as test_path
test_path = 'data/dev'

classes = [str(i) for i in range(1, 32)]

train_batches = ImageDataGenerator().flow_from_directory(train_path, 
    target_size=(80,80),
    classes=classes,
    batch_size=31)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path, 
    target_size=(80,80),
    classes=classes,
    batch_size=31)

test_batches = ImageDataGenerator().flow_from_directory(test_path, 
    target_size=(80,80),
    classes=classes,
    batch_size=62)


# example -- plot of a single train_batch
# imgs, labels = next(train_batches)
# plots(imgs, titles=labels)
# plt.show()


# BUILD CNN

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(Flatten())
model.add(Dense(31, activation='softmax'))

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# TRAINING OF CNN

model.fit_generator(train_batches, steps_per_epoch=6,
    validation_data=valid_batches,
    validation_steps=2, 
    epochs=4,
    verbose=2)


# PREDICTION on test_batches

def convert_labels(labels, num_labels=31):
    new = np.zeros(len(labels))
    for i, label in enumerate(labels):
        for j in range(num_labels):
            if label[j] == 1:
                new[i] = j + 1
    return new


test_imgs, test_labels = next(test_batches)

predictions = model.predict_generator(test_batches, steps=1, verbose=2)

# convert prediction labels and test labels
# so that confusion matrix is able to process it
pred = convert_labels(predictions)
test_lab = convert_labels(test_labels)

# np.set_printoptions(threshold=sys.maxsize)
# print(predictions)
# print(pred)


# PLOT CONFUSION MATRIX 

cm = confusion_matrix(test_lab, pred)

plot_confusion_matrix(cm, classes)
plt.show()



