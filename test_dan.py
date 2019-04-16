from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(80, 80, 3)))    # RGB (3) pictures 80×80 pixels
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))  # Shape is deduced from previous layer.
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()

train_path = 'data/train'
valid_path = 'data/dev'     # TODO valid_path is for now the same as test_path
test_path = 'data/dev'

classes = [str(i) for i in range(1, 32)]

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=(80, 80),
                                                         classes=classes,
                                                         batch_size=31)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=(80, 80),
                                                         classes=classes,
                                                         batch_size=31)

test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                        target_size=(80, 80),
                                                        classes=classes,
                                                        batch_size=62)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit_generator(train_batches, epochs=4, verbose=2)

test_loss, test_acc = model.evaluate(test_images, test_labels)print(‘Test accuracy:’, test_acc)