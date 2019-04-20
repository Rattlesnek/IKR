import warnings
import numpy as np
import keras
from keras import backend as K 
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os


def create_model():
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
    return model



def prepare_batches(train_path, valid_path, test_path):
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
    return train_batches, valid_batches, test_batches


def execute_training(model_path, weights_path):

    # PATH TO DATA
    train_path = 'data/data_pics_resized_224'
    valid_path = 'data/dev_pics_resized_224'
    test_path = 'data/dev_pics_resized_224'

    model = create_model()

    train_batches, valid_batches, test_batches = prepare_batches(train_path, valid_path, test_path)
    
    model.compile(Adam(lr=.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # TRAINING OF CNN
    model.fit_generator(train_batches, steps_per_epoch=16,
                        validation_data=valid_batches,
                        validation_steps=4, 
                        epochs=29, # best is 29 ... 93%
                        verbose=2)

    test_imgs, test_labels = next(test_batches)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(test_labels)
    test_loss = model.evaluate(test_imgs, test_labels, steps=2)

    print(model.metrics_names)
    print(test_loss)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_path)
    print("Saved model to", model_path, weights_path)



def get_filenames(*, path, suffix='.png'):
    """ Iterate through directories and yield classes and filenames """
    for fldr in sorted(os.listdir(path), key=lambda fldr: int(fldr)):
        clss = fldr
        folder = os.path.join(path, fldr)
        # check if it is a folder 
        if os.path.isdir(folder):            
            for fl in sorted(os.listdir(folder)):
                file = os.path.join(folder, fl)
                # check if it is a file with specified suffix 
                if os.path.isfile(file) and file.endswith(suffix):
                    yield clss, file


def execute_prediction(model_path, weights_path):
    
    test_path = 'data/eval_pics_resized_224'
    
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_path)
    print("Loaded model from", model_path, weights_path)

    predictions = []
    for ff in sorted(os.listdir(test_path)):
        file = os.path.join(test_path, ff)
        print(file)
        img = load_img(file, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0,:]
        predictions.append(np.log(pred))       
        
    return predictions

