import scipy.io
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import sys

######################## spam ###########################
#Read in and format training data
spam_train_raw = scipy.io.loadmat('hw01_data/spam/spam_data.mat')
#print scipy.io.whosmat('hw01_data/spam/spam_data.mat')
spam_train_features = spam_train_raw['training_data']
spam_train_labels = spam_train_raw['training_labels']

spam_train_labels = np.transpose(spam_train_labels)
#shuffle the rows
features_shuffled, labels_shuffled = shuffle(spam_train_features, spam_train_labels)
spam_all = np.hstack((features_shuffled, labels_shuffled))

#split into training and validation sets
spam_train = spam_all[:4000,:]
spam_validate = spam_all[4000:,:]

X_validate = spam_all[4000:,:-1]
y_validate = spam_all[4000:,-1]

#Learning curves
print "num_train train_score validation_score"
for num_train in (100, 200, 500, 1000, 2000, 4000):
    print num_train,
    X_train = spam_train[:num_train,:-1]
    y_train = spam_train[:num_train,-1]
    model = svm.SVC(kernel='linear').fit(X_train,y_train)
    print model.score(X_train,y_train),
    print model.score(X_validate,y_validate)

#Hyperparameter tests
X_train = spam_all[:,:-1]
y_train = spam_all[:,-1]  

print "5-fold cross validation"
for c in range(1,100):
    print c, 
    model = svm.SVC(C=c,kernel='linear').fit(X_train,y_train)
    print np.mean(cross_val_score(model,X_train,y_train,cv=5))

#Best model
print "training with C = 6"
X_train = spam_all[:,:-1]
y_train = spam_all[:,-1]
model = svm.SVC(kernel='linear').fit(X_train,y_train)

X_test = spam_train_raw['test_data']
prediction = model.predict(X_test)

print "Id,Category"
for i in range(len(prediction)):
    string = str(i) + "," + str(int(prediction[i]))
    sys.stdout.write(string)
    print ''




    





