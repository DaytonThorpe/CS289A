import scipy.io
from sklearn.utils import shuffle
import numpy as np

franzia_train_raw = scipy.io.loadmat('data.mat')
#print scipy.io.whosmat('data.mat')
features = franzia_train_raw['X']
labels = franzia_train_raw['y']

features_shuffled, labels_shuffled = shuffle(features, labels)

#Append a column of ones to the features
n = len(features)
d = len(features[0])

new_features = np.zeros([n,d+1])
new_features[:,:-1] = features_shuffled
new_features[:,-1] = 1.0
features_shuffled = new_features

n_total = n
d = len(features_shuffled[0])

w = np.ones([d,1])

Lambda = 0.1
rate_0 = 10.0

#center
for i in range(d-1): #don't center the bias feature
    mean = 0
    for j in range(n_total): mean += features_shuffled[j][i]
    mean /= n_total
    for j in range(n_total): features_shuffled[j][i] -= mean

#normalize
for i in range(d-1): #don't normalize the bias feature
    var = 0
    for j in range(n_total): var += features_shuffled[j][i] * features_shuffled[j][i]
    var /= n_total
    for j in range(n_total): features_shuffled[j][i] /= np.sqrt(var)

#split into training and test
features_test = features_shuffled[5000:,:]
labels_test = labels_shuffled[5000:,:]
features_shuffled = features_shuffled[:5000,:]
labels_shuffled = labels_shuffled[:5000,:]

n = len(features_shuffled)
s = np.zeros([n,1])

for i in range(1000):
    rate = rate_0 / (1+i)
    for j in range(n):
        s[j] = 1 / (1 + np.exp(-features_shuffled[j].dot(w)))
    grad = -(labels_shuffled[i] - s[i])*features_shuffled[i]
    #Don't apply regularization to bias
    for j in range(d-1): w[j] = w[j] - rate*(grad[j] + 2*Lambda*w[j])
    #bias
    w[d-1] = w[d-1] - rate*(grad[d-1])

    if i%10 != 0: continue
    cost = 0
    for j in range(n):
        if s[j] == 1: s[j] = 1 - 1e-10
        elif s[j] == 0: s[j] = 1e-10
        cost -= labels_shuffled[j]*np.log(s[j])
        #print "s: ", s[j]
        cost -= (1-labels_shuffled[j])*np.log(1-s[j])
    for j in range(d-1):
        cost += Lambda * w[j] * w[j]
    print i, cost[0]

#predictions
accuracy = 0
for i in range(1000):
    s_test = 1 / (1 + np.exp(-features_test[i].dot(w)))
    #print "features: ", features_test[i]
    #print "w: ", w
    #print "s, label: ", s_test, labels_test[i]
    #print s_test
    if s_test >= 0.5 and labels_test[i] == 1: accuracy += 1
    elif s_test < 0.5 and labels_test[i] == 0: accuracy += 1

accuracy /= 1000.0
print "accuracy: ", accuracy

