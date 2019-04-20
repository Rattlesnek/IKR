IKR - World's best classifier!


USAGE

usage: main.py [-h] [-t] [-p] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           train model and save it (speech classifier doesn't save model)
  -p, --predict         load model and predict
  -s, --spech		train model and predict for speech data
  -m MODEL, --model MODEL
                        specify model for training / predicting



MODELS

The model must be specified using "-m MODEL" or "--model MODEL". 
Instead of "MODEL" insert one of the following available models.

Face recognition models:
* VGG       - model created from VGG-Face model using "transfer learing" technique
* VGG+SVM   - model created using VGG-Face model as image encoder followed by SVM for classification
* HOG+SVM   - model created using HoG in combination with SVM



DEPENDENCIES

* python3         - for face recognition
* python2         - for speech recognition
* numpy           - pip install numpy
* scipy	          - pip install scipy
* OpenCV          - pip install opencv
* scikit-learn    - pip install scikit-learn
* Keras           - pip install keras
* joblib          - pip install joblib
* VGG-Face model weights - https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view
                         - paste it to the folder where the whole project is located
* ikrlib
* maybe others :)



GET WHOLE PROJECT

To run training and prediction download the whole project from this link:



