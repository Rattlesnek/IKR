import sys
import os
import numpy as np
import time
import collections

import vgg_face_model as vggf


def get_filenames(path='data/train_pics_resized_224', suffix='.png'):
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


def determine_class(arr):
    minimum = min(arr)
    for i in range(len(arr)):
        if arr[i] == minimum:
            return str(i + 1)


def print_min(arr):
    minimum = min(arr)
    print('minimum:', minimum)
    for i in range(len(arr)):
        if arr[i] == minimum:
            print('minimum class:', i + 1)


# database_path = 'data/try_train_224'
# test_path = 'data/try_dev_224'

database_path = 'data/train_pics_resized_224'
test_path = 'data/dev_pics_resized_224'

num_classes = 31

##############################################

# test_img = 'data/try_dev_224/1/f401_04_f16_i0_0.png'
# test_img_repr = vggf.get_img_representation(test_img)

# cosine = np.zeros(num_classes)
# euclid = np.zeros(num_classes)
# for clss, file in get_filenames(path=database_path):
#     idx = int(clss) - 1
#     print(clss, ':', file)
#     database_img_repr = vggf.get_img_representation(file)
#     cosine[idx] += vggf.findCosineSimilarity(database_img_repr, test_img_repr)
#     euclid[idx] += vggf.findEuclideanDistance(database_img_repr, test_img_repr)

###############################################

vgg_descriptor = vggf.create_model_descriptor()


start_time = time.time()

# LOAD DATABASE OF TRAINING DATA (train)
print('load database')

database_repr = collections.defaultdict(list)
for clss, file in get_filenames(path=database_path):
    print(clss, ':', file)
    database_repr[clss].append(vggf.get_img_representation(vgg_descriptor, file))


# EVALUATE TEST IMAGES (dev)
print('eval test imgs')

cosine_expected_actual = []
euclid_expected_actual = []
for test_clss, test_file in get_filenames(path=test_path):
    print(test_clss, ':', file)
    
    cosine = np.zeros(num_classes)
    euclid = np.zeros(num_classes)
    test_repr = vggf.get_img_representation(vgg_descriptor, test_file)
    for data_clss, data_repr_list in database_repr.items():
        idx = int(data_clss) - 1
        for data_repr in data_repr_list:
            cosine[idx] += vggf.findCosineSimilarity(data_repr, test_repr)
            euclid[idx] += vggf.findEuclideanDistance(data_repr, test_repr)
    
    cosine_actual_clss = determine_class(cosine)
    print('cosine', cosine_actual_clss)
    cosine_expected_actual.append((test_clss, cosine_actual_clss))
    
    euclid_actual_clss = determine_class(euclid)
    print('euclid', euclid_actual_clss)
    euclid_expected_actual.append((test_clss, euclid_actual_clss))


def num_correct(expected_actual):
    correct = 0
    for expected, actual in expected_actual:
        if expected == actual:
            correct += 1
    return correct

# metric no.1
print('\nCOSINE ACCURACY: ', (100.0 / len(cosine_expected_actual)) * num_correct(cosine_expected_actual), '\n')

# metric no.2
print('\nEUCLID ACCURACY: ', (100.0 / len(euclid_expected_actual)) * num_correct(euclid_expected_actual), '\n')


end_time = time.time()
print('eval time:', end_time - start_time)


