import glob
import numpy as np
import vgg_face_model as vfm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load


number_of_classes = 31


def load_folder(folder_name):
    model = vfm.create_model_descriptor()
    images = []
    for filename in glob.iglob('data/' + folder_name + '/**/*.png', recursive=True):
        vector = vfm.get_img_representation(model, filename)
        images.append(vector)
    images = np.array(images)
    return images


def load_data():
    images1 = load_folder("train_pics_resized_224")
    labels1 = np.repeat(np.arange(number_of_classes) + 1, len(images1) / number_of_classes)
    images2 = load_folder("dev_pics_resized_224")
    labels2 = np.repeat(np.arange(number_of_classes) + 1, len(images2) / number_of_classes)
    all_images = np.concatenate((images1, images2), axis=0)
    all_labels = np.concatenate((labels1, labels2), axis=0)
    rand = np.random.RandomState(number_of_classes)
    shuffle = rand.permutation(len(all_images))
    all_images, all_labels = all_images[shuffle], all_labels[shuffle]
    return all_images, all_labels


def train_model():
    images, labels = load_data()
    classifier = SVC(gamma='scale', C=10, kernel='rbf', probability=True)
    classifier.fit(images, labels)
    dump(classifier, 'models/svm_vgg.joblib')


def test_model():
    images, labels = load_data()
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.20)
    classifier = SVC(gamma='scale', C=10, kernel='rbf', probability=True)
    classifier.fit(train_images, train_labels)
    confidence = classifier.score(test_images, test_labels)
    predictions = classifier.predict(test_images)
    print("Accuracy: " + str(confidence))
    print(classification_report(test_labels, predictions))


def predict_data(model_path):
    test_images = load_folder('eval_pics_resized_224')
    classifier = load(model_path)
    return classifier.predict_log_proba(test_images)
