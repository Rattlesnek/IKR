import warnings
import numpy as np
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
warnings.simplefilter(action='ignore', category=FutureWarning)

import confusion_matrix as cnf_m



model_vgg = Sequential()
model_vgg.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model_vgg.add(Convolution2D(64, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(64, (3, 3), activation='relu'))
model_vgg.add(MaxPooling2D((2,2), strides=(2,2)))

model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(128, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(128, (3, 3), activation='relu'))
model_vgg.add(MaxPooling2D((2,2), strides=(2,2)))

model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(256, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(256, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(256, (3, 3), activation='relu'))
model_vgg.add(MaxPooling2D((2,2), strides=(2,2)))

model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(MaxPooling2D((2,2), strides=(2,2)))

model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(ZeroPadding2D((1,1)))
model_vgg.add(Convolution2D(512, (3, 3), activation='relu'))
model_vgg.add(MaxPooling2D((2,2), strides=(2,2)))

model_vgg.add(Convolution2D(4096, (7, 7), activation='relu'))
model_vgg.add(Dropout(0.5))
model_vgg.add(Convolution2D(4096, (1, 1), activation='relu'))
model_vgg.add(Dropout(0.5))
model_vgg.add(Convolution2D(2622, (1, 1)))
model_vgg.add(Flatten())
model_vgg.add(Activation('softmax'))

model_vgg.load_weights('vgg_face_weights.h5')
# model = model_vgg

model = Sequential()
for layer in model_vgg.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(31, activation='softmax'))

print(model.summary())



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


model.compile(Adam(lr=.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# TRAINING OF CNN
model.fit_generator(train_batches, steps_per_epoch=6,
    validation_data=valid_batches,
    validation_steps=2, 
    epochs=22,
    verbose=2)


