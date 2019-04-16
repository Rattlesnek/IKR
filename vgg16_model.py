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


# PATH TO DATA
train_path = 'data/train_pics_resized_224'
valid_path = 'data/dev_pics_resized_224' # TODO valid_path is for now the same as test_path
test_path = 'data/dev_pics_resized_224'


classes = [str(i) for i in range(1, 32)]

train_batches = ImageDataGenerator().flow_from_directory(train_path, 
    target_size=(224,224),
    classes=classes,
    batch_size=31)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path, 
    target_size=(224,224),
    classes=classes,
    batch_size=31)

test_batches = ImageDataGenerator().flow_from_directory(test_path, 
    target_size=(224,224),
    classes=classes,
    batch_size=62)


# BUILD CNN from VGG16 model

vgg16_model = keras.applications.vgg16.VGG16()

# create sequential model with layers from VGG16
model = Sequential()
for i, layer in enumerate(vgg16_model.layers):
    # remove last layer Dense() -- classification into 1000 categories
    if i != 22:
        model.add(layer)

# forbid training of layers
for layer in model.layers:
    layer.trainable = False

# add new last layer -- classification into 31 categories
model.add(Dense(31, activation='softmax'))

print(model.summary())


model.compile(Adam(lr=.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# TRAINING OF CNN

model.fit_generator(train_batches, steps_per_epoch=6,
    validation_data=valid_batches,
    validation_steps=2, 
    epochs=14,
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



