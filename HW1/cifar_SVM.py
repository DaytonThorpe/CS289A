import scipy.io
import random
from sklearn import svm

########### CIFAR ######################################################################
#Read in train data
cifar_train_raw = scipy.io.loadmat('hw01_data/cifar/train.mat')
cifar_all = cifar_train_raw['trainX']
random.shuffle(cifar_all) #Shuffle the data into training and validation sets

cifar_train = cifar_all[:45000,:]
cifar_validate = cifar_all[45000:,:]

X_validate = cifar_validate[:,:-1]
y_validate = cifar_validate[:,-1]

#Learning curves
print "num_train train_score validation_score"
for num_train in (100, 200, 500, 1000, 2000, 5000):
    print num_train,
    X_train = cifar_train[:num_train,:-1]
    y_train = cifar_train[:num_train,-1]
    model = svm.SVC(kernel='linear').fit(X_train,y_train)
    print model.score(X_train,y_train),
    print model.score(X_validate,y_validate)
