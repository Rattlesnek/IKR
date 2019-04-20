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
    dump(classifier, 'models/svm_hog.joblib')


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
    hist_of_grad = create_hog()
    hog_descriptors = get_hog(hist_of_grad, test_images)
    classifier = load(model_path)
    return classifier.predict_log_proba(hog_descriptors)


"""
OpenCV SVM with nice evaluation of results but without probabilities :(.


common.py:
'''
This module contains some common routines used by other samples.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2

# built-in modules
import os
import itertools as it
from contextlib import contextmanager

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

class Bunch(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __str__(self):
        return str(self.__dict__)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def homotrans(H, x, y):
    xs = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    ys = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    s  = H[2, 0]*x + H[2, 1]*y + H[2, 2]
    return xs/s, ys/s

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx,  0, tx],
                    [  0, cy, ty],
                    [  0,  0,  1]])
    return M


def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv2.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)

def nothing(*arg, **kw):
    pass

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    '''Make a grid from images.
    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    if PY3:
        img0 = next(imgs)
    else:
        img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, color)
            
svm_opencv.py:
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
