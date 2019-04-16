from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
# RGB (3) pictures 80×80 pixels
model.add(layers.Conv2D(32, (5, 5), activation='relu', data_format='channels_last', input_shape=(80, 80, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))  # Shape is deduced from previous layer.
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(31, activation='softmax'))
model.summary()

# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (5, 5), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
#
# train_path = 'data/train'
# valid_path = 'data/dev'     # TODO valid_path is for now the same as test_path
# test_path = 'data/dev'
#
# classes = [str(i) for i in range(1, 32)]
#
# data_generator = ImageDataGenerator(rotation_range=20,
#                                     horizontal_flip=True,
#                                     fill_mode='nearest')
#
# train_batches = data_generator.flow_from_directory(train_path,
#                                                    target_size=(80, 80),
#                                                    classes=classes,
#                                                    batch_size=31)
#
# valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
#                                                          target_size=(80, 80),
#                                                          classes=classes,
#                                                          batch_size=31)
#
# test_batches = ImageDataGenerator().flow_from_directory(test_path,
#                                                         target_size=(80, 80),
#                                                         classes=classes,
#                                                         batch_size=62)
#
# model.fit_generator(train_batches,
#                     steps_per_epoch=6,
#                     epochs=5,
#                     validation_data=valid_batches,
#                     validation_steps=2,
#                     verbose=2)

# predictions = model.predict_generator(test_batches, steps=1, verbose=2)
# print(predictions)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          batch_size=100,
          epochs=5,
          verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)