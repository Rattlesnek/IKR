import os
import cv2
import numpy as np
from common import mosaic


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*80*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (80, 80), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


train_images = []
for image_class in range(1, 32):
    directory_in_str = 'data/train/' + str(image_class) + '/'
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            file = directory_in_str + filename
            train_images.append(cv2.imread(file, 0))
train_labels = np.repeat(np.arange(31)+1, len(train_images)/31)
train_images = np.array(train_images)


"""
Train on horizontally flipped pictures too.
train_images2 = []
for image_class in range(1, 32):
    directory_in_str = 'data/train_pics_flipped_horizontally/' + str(image_class) + '/'
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            file = directory_in_str + filename
            train_images2.append(cv2.imread(file, 0))
np.append(train_labels, np.repeat(np.arange(31)+1, len(train_images)/31))
np.append(train_images, np.array(train_images2))
"""

test_images = []
for image_class in range(1, 32):
    directory_in_str = 'data/dev/' + str(image_class) + '/'
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            file = directory_in_str + filename
            test_images.append(cv2.imread(file, 0))
test_labels = np.repeat(np.arange(31)+1, len(test_images)/31)
test_images = np.array(test_images)

"""
Shuffling the training data - didn't help.
rand = np.random.RandomState(31)
shuffle = rand.permutation(len(train_images))
train_images, train_labels = train_images[shuffle], train_labels[shuffle]
"""

winSize = (80, 80)      # Image size.
blockSize = (40, 40)    # 2× cellSize, but try also other values.
blockStride = (40, 40)  # Typically half of blockSize.
cellSize = (10, 10)     # Maybe higher? Give it a try.
nbins = 24              # Can be increased (e. g. to 18), but 9 is recommended.
derivAperture = 1
winSigma = -1.0
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True  # Can be also False, try both.

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

train_images = list(map(deskew, train_images))

hog_descriptors_train = []
for image in train_images:
    hog_descriptors_train.append(hog.compute(image))
hog_descriptors_train = np.squeeze(hog_descriptors_train)

hog_descriptors_test = []
for image in test_images:
    hog_descriptors_test.append(hog.compute(image))
hog_descriptors_test = np.squeeze(hog_descriptors_test)

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
