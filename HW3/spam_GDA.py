import numpy as np
from sklearn.utils import shuffle
import scipy.io
from numpy import linalg as LA
import sys

#Read in training data
spam_train_raw = scipy.io.loadmat('hw3_spam_dist/spam_data.mat')
#print scipy.io.whosmat('hw3_spam_dist/spam_data.mat')
spam_train_features = spam_train_raw['training_data']
spam_train_labels = spam_train_raw['training_labels'].transpose()

#print "len(spam_train_features) ", len(spam_train_features)


#shuffle the rows
features_shuffled, labels_shuffled = shuffle(spam_train_features, spam_train_labels)
spam_all = np.hstack((features_shuffled, labels_shuffled))

spam_validate = spam_all[10000:,:]


#select a small subset of the data for easier processing
for size in (100, 1000, 5000, 10000, len(spam_train_features)):
    if size < 15000: continue
    spam_subset = spam_all[:size,:]
    num_features = len(spam_subset[0]) - 1

    #normalize the vectors
    for i in range(len(spam_subset)):
        l2_norm = 0
        for j in range(num_features):
            l2_norm += spam_subset[i,j] * spam_subset[i,j]
        l2_norm = np.sqrt(l2_norm)
        if l2_norm > 0:
            for j in range(num_features):
                spam_subset[i,j] = spam_subset[i,j] / l2_norm

    #calculate the means
    mu = np.zeros([2,num_features])
    count = [0]*2
    for i in range(size):
        indicator = spam_subset[i,-1]
        #print "indicator ", indicator
        for j in range(num_features): mu[int(indicator)][j] += spam_subset[i,j]
        count[int(indicator)] += 1
    #print "count: ", count
    for i in range(2):
        for j in range(num_features): mu[i][j] /= count[i]
    

    #calculate the covariance matrices
    Cov_QDA = [None]*2
    for i in range(2):
        #pick every row of digit_all such that the last column is equal to the current digit class
        Cov_QDA[i] = np.cov(spam_subset[spam_subset[:,-1]==i,:-1].transpose())
    #print "dimenion of normed_digit: ", len(normed_digit[0])
    #print "dimension: ", len(Cov[0]), len(Cov[0][0])
    #print digit_all[digit_all[:,-1]==1,:-1]
    #print "normed_digit: ", normed_digit[normed_digit[:,-1]==3,-1]
    

    #######  LDA  #######
    Cov_LDA = Cov_QDA[0]*count[0]
    for i in range(1,2):
        Cov_LDA += Cov_QDA[i]*count[i]
    Cov_LDA /= size
    #Add a perturbation along the diagonal to make the matrix non-singular
    for i in range(len(Cov_LDA)): Cov_LDA[i,i] += 1e-4
    CovInv = LA.inv(Cov_LDA)
    
    num_right = 0
    for i in range(10000):
        LDF = -1e20
        x = spam_validate[i,:-1]
        this_message = -1
        for j in range(2):
            LDF_new = mu[j].dot(CovInv.dot(x)) - 0.5 * mu[j].dot(CovInv.dot(mu[j]))
            if LDF_new > LDF:
                LDF = LDF_new
                this_message = j
        #print "LDF = ", LDF, "this_digit = ", this_message
        if this_message == spam_validate[i,-1]: num_right += 1
        elif this_message == -1:
            print "Error! No Prediction made!"
            sys.exit()
    accuracy = num_right / 10000.0
    print "For training on ", size, "num_right is ", num_right, " samples, accuracy is ", accuracy
    
    """
    #####  QDA #####
    CovInv_QDA = [None]*2
    for j in range(2):
        for i in range(len(Cov_QDA[j])): Cov_QDA[j][i,i] += 1e-4
    for i in range(2): CovInv_QDA[i] = LA.inv(Cov_QDA[i])
    #Calculate the log determinant as the sum of the log of the eigenvalues
    log_det = [0]*2
    for j in range(2):
        lambda_v, v = LA.eig(CovInv_QDA[j])
        log_lambda = np.log(lambda_v)
        log_det[j] = np.sum(log_lambda)
    
    num_right = 0
    for i in range(10000):
        QDF = -1e20
        x = spam_validate[i,:-1]
        this_message = -1
        for j in range(2):
            QDF_new = - (x - mu[j]).transpose().dot(CovInv_QDA[j].dot(x-mu[j]))
            QDF_new -= log_det[j]
            if QDF_new > QDF:
                QDF = QDF_new
                this_message = j
        if this_message == spam_validate[i,-1]: num_right += 1
    accuracy = num_right / float(len(spam_validate))
    print "For training QDA on ", size, "num_right is ", num_right, " samples, accuracy is ", accuracy
    """
    spam_test_raw = scipy.io.loadmat('hw3_spam_dist/spam_data.mat')
    spam_test = spam_test_raw['test_data']
    #print "length of digit test: ", len(digit_test)
    #print "number of pixels: ", len(digit_test[0])
    print "Id,Category"
    for i in range(len(spam_test)):
        LDF = -1e20
        x = spam_test[i,:]
        this_message = -1
        for j in range(2):
            LDF_new = mu[j].dot(CovInv.dot(x)) - 0.5 * mu[j].dot(CovInv.dot(mu[j]))
            if LDF_new > LDF:
                LDF = LDF_new
                this_message = j
        string = str(i) + "," + str(this_message)
        sys.stdout.write(string)
        print ''











