from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gmm, logpdf_gmm, logistic_sigmoid
import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
from numpy.random import randint
from itertools import cycle

N_CLASSES = 31
N_TESTS = 2
testing = False

if testing:
    train = [wav16khz2mfcc("data/train/" + str(i)).values() for i in range(1, N_CLASSES + 1)]
    test = [wav16khz2mfcc("data/dev/" + str(i)).values() for i in range(1,N_CLASSES + 1)]

else:
    train = [wav16khz2mfcc("data/train/" + str(i)).values() for i in range(1, N_CLASSES + 1)]
    test = [wav16khz2mfcc("data/dev/" + str(i)).values() for i in range(1,N_CLASSES + 1)]
    train = train + test
    Eval = wav16khz2mfcc("data/eval")

orig_train = np.copy(train)

for i in range(0,N_CLASSES):
    train[i] = np.vstack(train[i])

dim = train[1][0].shape[0]

#GMM parameters
n_comps = [30 for i in range(N_CLASSES)]

mus = [train[i][randint(1,len(train[i]),n_comps[i])] for i in range(N_CLASSES)]

covs = [[np.var(train[i], axis=0)] * n_comps[i] for i in range(N_CLASSES)]

weights = [np.ones(n_comps[i])/n_comps[i] for i in range(N_CLASSES)]


priors = [1.0/N_CLASSES for i in range(0, N_CLASSES)]

TTL = [0 for i in range(N_CLASSES)]

#EMM
for i in range(100):
    print("Iteration: ", i)
    for c in range(N_CLASSES):
        [weights[c], mus[c], covs[c], TTL[c]] = train_gmm(train[c], weights[c], mus[c], covs[c])
        print('\t Total log-likelihood:' + str(TTL[c]))

#Test mode
if testing:
    #Evaluate on train data
    #train_ok_count = 0
    #for i in range(N_CLASSES):
    #    for x in orig_train[i]:
    #        logpdfs = [0 for j in range(N_CLASSES)]
    #        for c in range(N_CLASSES):
    #            logpdfs[c] = logpdf_gmm(x, weights[c], mus[c], covs[c])
    #            logpdfs[c] = sum(logpdfs[c]) + np.log(priors[c])
    #        res = np.argmin(logpdfs)
    #
    #        if res == i:
    #            train_ok_count+=1

    #Evaluate on test data
    ok_count = 0
    for i in range(N_CLASSES):
        for x in test[i]:
            score = [0 for j in range(N_CLASSES)]
            logpdfs = [0 for j in range(N_CLASSES)]
            for c in range(N_CLASSES):
                logpdfs[c] = logpdf_gmm(x, weights[c], mus[c], covs[c])
                score[c] = (sum(logpdfs[c]) + np.log(priors[c]))
            res = np.argmax(score)

            if res == i:
                print("Good")
                ok_count+=1
            else:
                print("Wrong, result was " + str(res) + " instead of " + str(i))

    #print("Train score: ", (float(train_ok_count)/186)*100)
    print("Test score: ", (float(ok_count)/62)*100)


#Evaluation mode
else:
    print("not testing")
    filename = "eval/eval_score_speech_gmm.txt"
    f = open(filename, "w+")

    ok_count = 0
    for key, x in sorted(Eval.iteritems()):
        score = [0 for j in range(N_CLASSES)]
        logpdfs = [0 for j in range(N_CLASSES)]
        for c in range(N_CLASSES):
            logpdfs[c] = logpdf_gmm(Eval[key], weights[c], mus[c], covs[c])
            score[c] = (sum(logpdfs[c]) + np.log(priors[c]))
        res = np.argmax(score)

        name = key.replace("data/eval/","")
        name = name.replace(".wav","")
        vals = str(score).replace("[","")
        vals = vals.replace("]","")
        vals = vals.replace(",","")
        f.write(name + " " + str(res+1) + " " + vals + "\n")

