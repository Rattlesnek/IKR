import glob
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load


number_of_classes = 31


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    new_m = np.float32([[1, skew, -0.5*80*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, new_m, (80, 80), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def load_folder(folder_name):
    images = []
    for file_name in glob.iglob('data/' + folder_name + '/**/*.png', recursive=True):
        images.append(cv2.imread(file_name, 0))
    images = np.array(images)
    images = list(map(deskew, images))
    return images


def load_data():
    images1 = load_folder("train")
    labels1 = np.repeat(np.arange(number_of_classes) + 1, len(images1) / number_of_classes)
    images2 = load_folder("dev")
    labels2 = np.repeat(np.arange(number_of_classes) + 1, len(images2) / number_of_classes)
    all_images = np.concatenate((images1, images2), axis=0)
    all_labels = np.concatenate((labels1, labels2), axis=0)
    rand = np.random.RandomState(number_of_classes)
    shuffle = rand.permutation(len(all_images))
    all_images, all_labels = all_images[shuffle], all_labels[shuffle]
    hist_of_grad = create_hog()
    hog_descriptors = get_hog(hist_of_grad, all_images)
    return hog_descriptors, all_labels


def create_hog():
    win_size = (80, 80)  # Image size.
    block_size = (40, 40)  # 2× cellSize, but try also other values.
    block_stride = (40, 40)  # Typically half of blockSize.
    cell_size = (10, 10)  # Maybe higher? Give it a try.
    n_bins = 24  # Can be increased (e. g. to 18), but 9 is recommended.
    deriv_aperture = 1
    win_sigma = -1.0
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    n_levels = 64
    signed_gradients = True  # Can be also False, try both.
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins, deriv_aperture, win_sigma,
                             histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels, signed_gradients)


def get_hog(hog, images):
    hog_descriptors = []
    for image in images:
        hog_descriptors.append(hog.compute(image))
    return np.squeeze(hog_descriptors)


def train_model():
    hog_descriptors, labels = load_data()
    classifier = SVC(C=12.5, kernel='linear', probability=True)
    classifier.fit(hog_descriptors, labels)
    dump(classifier, 'models/hog_svm.joblib')


def test_model():
    hog_descriptors, labels = load_data()
    train_hog, test_hog, train_labels, test_labels = train_test_split(hog_descriptors, labels, test_size=0.20)
    classifier = SVC(C=12.5, kernel='linear', probability=True)
    classifier.fit(train_hog, train_labels)
    confidence = classifier.score(test_hog, test_labels)
    predictions = classifier.predict(test_hog)
    print("Accuracy: " + str(confidence))
    print(classification_report(test_labels, predictions))


def predict_data(model_path):
    test_images = load_folder('eval')
    classifier = load(model_path)
    return classifier.predict_log_proba(test_images)


"""
OpenCV SVM with nice evaluation of results but without probabilities :(.

from common import mosaic

C = 12.5
gamma = 0.50625
svm = cv2.ml.SVM_create()
svm.setGamma(gamma)
svm.setC(C)
svm.setKernel(cv2.ml.SVM_INTER)
svm.setType(cv2.ml.SVM_NU_SVC)
svm.setNu(0.9)

# Train SVM on training data
svm.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, train_labels)

# Test on a held out test set
predictions = svm.predict(hog_descriptors_test)[1].ravel()
accuracy = (test_labels == predictions).mean()
print('Percentage Accuracy: %.2f %%' % (accuracy * 100))

confusion = np.zeros((31, 31), np.int32)
for i, j in zip(test_labels, predictions):
    confusion[int(i)-1, int(j)-1] += 1
print('confusion matrix:')
print(confusion)

vis = []
for img, flag in zip(test_images, predictions == test_labels):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not flag:
        img[..., :2] = 0
    vis.append(img)
vis = mosaic(25, vis)

cv2.imwrite("digits-classification.jpg", vis)
cv2.imshow("Vis", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
