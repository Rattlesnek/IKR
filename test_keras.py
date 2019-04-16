import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import confusion_matrix as cnf_m

import sys


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
#
# ONLY EXAMPLE !!! (classification is bad)
#

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(Flatten())
model.add(Dense(31, activation='softmax'))

print(model.summary())

model.compile(Adam(lr=.0001), 
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# TRAINING OF CNN

model.fit_generator(train_batches, steps_per_epoch=6,
    validation_data=valid_batches,
    validation_steps=2, 
    epochs=5,
    verbose=2)


# PREDICTION on test_batches

predictions = model.predict_generator(test_batches, steps=1, verbose=2)

test_imgs, test_labels = next(test_batches)

# convert prediction labels and test labels
# so that confusion matrix is able to process it
pred = cnf_m.convert_labels(predictions)
test_lab = cnf_m.convert_labels(test_labels)

# np.set_printoptions(threshold=sys.maxsize)
# print(predictions)
# print(pred)


# PLOT CONFUSION MATRIX 

cm = confusion_matrix(test_lab, pred)

cnf_m.plot_confusion_matrix(cm, classes)
plt.show()

