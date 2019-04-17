import os
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline


train_images = []
for image_class in range(1, 32):
    directory_in_str = 'data/np_train/' + str(image_class) + '/'
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            file = directory_in_str + filename
            train_images.append(np.load(file))
train_labels = np.repeat(np.arange(31)+1, len(train_images)/31)
train_images = np.array(train_images)
