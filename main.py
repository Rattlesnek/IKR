import sys
import argparse

import vgg_face_train as vft


def training(model_name):
    
    if model_name.upper() == 'VGG':
        print('Using overtrained VGG')

        model_path = 'models/vgg_overtrained.json'
        weights_path = 'models/vgg_overtrained.h5'
        vft.execute_training(model_path, weights_path)

    elif model_name.upper() == 'VGG+SVM':
        print('Using VGG + SVM')
        
        ...

    elif model_name.upper() == 'SVM':
        print('Using HoG + SVM')

        ...

    else:
        print('ERROR: Unrecognized model', model_name, file=sys.stderr)
        sys.exit(1)



def prediction(model_name):
    
    if model_name.upper() == 'VGG':
        print('Using overtrained VGG')

        model_path = 'models/vgg_overtrained.json'
        weights_path = 'models/vgg_overtrained.h5'
        predict = vft.execute_prediction(model_path, weights_path)
        print(predict)


    elif model_name.upper() == 'VGG+SVM':
        print('Using VGG + SVM')
        
        ...

    elif model_name.upper() == 'SVM':
        print('Using HoG + SVM')

        ...

    else:
        print('ERROR: Unrecognized model', model_name, file=sys.stderr)
        sys.exit(1)




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
        # PREDICt
        prediction(args.model)

