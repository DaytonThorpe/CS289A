import scipy.io
from sklearn import svm
import random
import sys

########### MNIST   ######################################################################
#Read in train data
digit_train_raw = scipy.io.loadmat('hw01_data/mnist/train.mat')
digit_all = digit_train_raw['trainX']
random.shuffle(digit_all) #Shuffle the data into training and validation sets

digit_train = digit_all[:50000,:]
digit_validate = digit_all[50000:,:]

X_validate = digit_validate[:,:-1]
y_validate = digit_validate[:,-1]

#Learning curves
print "num_train train_score validation_score"
for num_train in (100, 200, 500, 1000, 2000, 5000, 10000):
    print num_train,
    X_train = digit_train[:num_train,:-1]
    y_train = digit_train[:num_train,-1]
    model = svm.SVC(kernel='linear').fit(X_train,y_train)
    print model.score(X_train,y_train),
    print model.score(X_validate,y_validate)

#Hyperparameter tests
X_train = digit_train[:10000,:-1]
y_train = digit_train[:10000,-1]
for c in range(1,10):
    print c*pow(10,-7),
    model = svm.SVC(C=c*pow(10,-7),kernel='linear').fit(X_train,y_train)
    print model.score(X_train,y_train),
    print model.score(X_validate,y_validate)
    #print pow(10,-c),
    #model = svm.SVC(C=pow(10,-c),kernel='linear').fit(X_train,y_train)
    #print model.score(X_train,y_train),
    #print model.score(X_validate,y_validate)


#Best model
print "Training with 50,000 examples and C = 6e-7."
X_train = digit_train[:50000,:-1]
y_train = digit_train[:50000,-1]
model = svm.SVC(C=6*pow(10,-7),kernel='linear').fit(X_train,y_train)
print model.score(X_train,y_train),
print model.score(X_validate,y_validate)

#Read in test data
digit_test_raw = scipy.io.loadmat('hw01_data/mnist/test.mat')
#print scipy.io.whosmat('hw01_data/mnist/test.mat')
digit_test = digit_test_raw['testX']

print "Id,Category"
prediction = model.predict(digit_test)
for i in range(len(prediction)):
    string = str(i) + "," + str(prediction[i])
    sys.stdout.write(string)
    print ''
