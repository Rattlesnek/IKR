import sys
import argparse
import numpy as np

import vgg_face_train as vft
import svm_vgg
import svm_hog


def training(model_name):
    
    if model_name.upper() == 'VGG':
        print('Using overtrained VGG')
        model_path = 'models/vgg_overtrained.json'
        weights_path = 'models/vgg_overtrained.h5'
        vft.execute_training(model_path, weights_path)

    elif model_name.upper() == 'VGG+SVM':
        print('Using VGG + SVM')
        svm_vgg.train_model()

    elif model_name.upper() == 'SVM':
        print('Using HoG + SVM')
        svm_hog.train_model()

    else:
        print('ERROR: Unrecognized model', model_name, file=sys.stderr)
        sys.exit(1)


def combine_results(image_results, audio_results):
    combined_results = []
    for index in range(0, len(image_results)):
        combined_results.append(image_results[index] + audio_results[index] -
                                image_results[index] * audio_results[index])
    return combined_results


def print_results(score):
    output = open("eval_score.txt", "w")
    indices = np.argmax(score, axis=1)
    for index in range(0, len(score)):
        output.write('eval_' + "{:05d}".format(index+1) + ' ' +
                     str(indices[index]+1) + ' ' +
                     ' '.join([str(x) for x in score[index].tolist()]) + '\n')
    output.close()


def prediction(model_name):
    if model_name.upper() == 'VGG':
        print('Using overtrained VGG')
        model_path = 'models/vgg_overtrained.json'
        weights_path = 'models/vgg_overtrained.h5'
        predict = vft.execute_prediction(model_path, weights_path)

    elif model_name.upper() == 'VGG+SVM':
        print('Using VGG + SVM')
        model_path = 'models/vgg_svm.joblib'
        predict = svm_vgg.predict_data(model_path)

    elif model_name.upper() == 'SVM':
        print('Using HoG + SVM')
        model_path = 'models/hog_svm.joblib'
        predict = svm_hog.predict_data(model_path)

    else:
        print('ERROR: Unrecognized model', model_name, file=sys.stderr)
        sys.exit(1)

    return predict


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train model and save it')
    parser.add_argument('-p', '--predict', action='store_true', help='load model and predict')
    parser.add_argument('-m', '--model', help='specify model for training / predicting')
    args = parser.parse_args()
    
    if not args.model:
        print('ERROR: Model not specified: use --model (-m)', file=sys.stderr)
        print('       Available models: vgg, vgg+svm, svm', file=sys.stderr)
        sys.exit(1)

    if not (args.train or args.predict) or (args.train and args.predict):
        print('ERROR: Need to specify only one argument: --train (-t) / --predict (-p)', file=sys.stderr)
        sys.exit(1)    

    if args.train:
        # TRAIN
        training(args.model)
    elif args.predict:
        # PREDICT
        results = prediction(args.model)
        print_results(results)


