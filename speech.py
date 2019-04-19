from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gmm, logpdf_gmm
import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
from numpy.random import randint
from itertools import cycle

N_CLASSES = 31

train = [wav16khz2mfcc("data/train/" + str(i)).values() for i in range(1, N_CLASSES + 1)]
test = [wav16khz2mfcc("data/dev/" + str(i)).values() for i in range(1,N_CLASSES + 1)]
orig_train = np.copy(train)


for i in range(0,N_CLASSES):
    train[i] = np.vstack(train[i])

dim = train[1][0].shape[0]

#GMM parameters
n_comps = [10 for i in range(N_CLASSES)]

mus = [train[i][randint(1,len(train[i]),n_comps[i])] for i in range(N_CLASSES)]

covs = [[np.var(train[i], axis=0)] * n_comps[i] for i in range(N_CLASSES)]

weights = [np.ones(n_comps[i])/n_comps[i] for i in range(N_CLASSES)]


priors = [1.0/N_CLASSES for i in range(0, N_CLASSES)]

TTL = [0 for i in range(N_CLASSES)]

#EMM
for i in range(40):
    print("Iteration: ", i)
    for c in range(N_CLASSES):
        [weights[c], mus[c], covs[c], TTL[c]] = train_gmm(train[c], weights[c], mus[c], covs[c])
        print('\t Total log-likelyhood:', TTL[c])

#Evaluate on train data
train_ok_count = 0
for i in range(N_CLASSES):
    for x in orig_train[i]:
        logpdfs = [0 for j in range(N_CLASSES)]
        for c in range(N_CLASSES):
            logpdfs[c] = logpdf_gmm(x, weights[c], mus[c], covs[c])
            logpdfs[c] = sum(logpdfs[c]) + np.log(priors[c])
        res = np.argmax(logpdfs)
        #eval
        if res == i:
            train_ok_count+=1

#Evaluate on test data
ok_count = 0
for i in range(N_CLASSES):
    for x in test[i]:
        logpdfs = [0 for j in range(N_CLASSES)]
        for c in range(N_CLASSES):
            logpdfs[c] = logpdf_gmm(x, weights[c], mus[c], covs[c])
            logpdfs[c] = sum(logpdfs[c]) + np.log(priors[c])
        res = np.argmax(logpdfs)
        hist.append(res)
        #eval
        if res == i:
            print("Good")
            ok_count+=1
        else:
            print("Bad, result was " + str(res) + " insted of " + str(i))

print("Train score: ", float(train_ok_count)/(31*len(orig_train[i]))*100)
print("Test score: ", float(ok_count)/(31*len(test[i]))*100)


