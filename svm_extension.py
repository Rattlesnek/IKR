import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.metrics import classification_report
from joblib import dump, load

number_of_classes = 31


def load_train_images(folder_name):
    images = []
    for image_class in range(1, number_of_classes+1):
        directory_in_str = 'data/' + folder_name + '/' + str(image_class) + '/'
        directory = os.fsencode(directory_in_str)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".npy"):
                file = directory_in_str + filename
                images.append(np.load(file))
    return np.array(images), np.repeat(np.arange(number_of_classes)+1, len(images)/number_of_classes)


def load_test_images(folder_name):
    images = []
    directory_in_str = folder_name + '/'
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            file = directory_in_str + filename
            images.append(np.load(file))
    return np.array(images)


def process_train_data():
    images1, labels1 = load_train_images("np_train")
    images2, labels2 = load_train_images("np_dev")
    all_images = np.concatenate((images1, images2), axis=0)
    all_labels = np.concatenate((labels1, labels2), axis=0)
    rand = np.random.RandomState(number_of_classes)
    shuffle = rand.permutation(len(all_images))
    all_images, all_labels = all_images[shuffle], all_labels[shuffle]
    return train_test_split(all_images, all_labels, test_size=0.50)


def train_model():
    train_images, test_images, train_labels, test_labels = process_train_data()
    classifier = SVC(gamma='scale', C=10, kernel='rbf', probability=True)
    classifier.fit(train_images, train_labels)
    dump(classifier, 'svm_extension.joblib')
    # confidence = classifier.score(test_images, test_labels)
    # predictions = classifier.predict(test_images)
    # print("Accuracy: " + str(confidence))
    # print(classification_report(test_labels, predictions))


def evaluate_data():
    test_images = load_test_images('test')
    classifier = load('svm_extension.joblib')
    return classifier.predict_log_proba(test_images)


print(evaluate_data())
