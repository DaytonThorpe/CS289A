import numpy as np
import random
import scipy.io
from numpy import linalg as LA
#import matplotlib.pyplot as plt
import sys

#Read in training data
digit_train_raw = scipy.io.loadmat('hw3_mnist_dist/train.mat')
digits = digit_train_raw['trainX']
random.shuffle(digits) #Shuffle the data into training and validation sets
digit_validate = digits[50000:,:]

#select a small subset of the data for easier processing
#for size in (100, 200, 500, 1000, 2000, 5000, 10000, 50000):
for size in (30000, 50000):
    #if size < 45000: continue
    digit_all = digits[:size,:]
    num_pixels = len(digit_all[0]) - 1
    #contrast normalize
    #normed_digit = [[0.0 for _ in range(num_pixels+1)] for _ in range(len(digit_all))]
    normed_digit = np.empty([len(digit_all), num_pixels+1])
    for i in range(len(digit_all)):
        l2_norm = 0
        for j in range(num_pixels): l2_norm += digit_all[i][j] * digit_all[i][j]
        l2_norm = np.sqrt(l2_norm)
        for j in range(num_pixels):
            normed_digit[i,j] = digit_all[i][j] / l2_norm
        normed_digit[i,num_pixels] = digit_all[i][-1]

    #calculate the means
    mu = np.zeros([10,num_pixels])
    count = [0]*10
    for i in range(len(normed_digit)):
        digit = normed_digit[i,-1]
        #print "digit: ", digit
        for j in range(num_pixels): mu[int(digit)][j] += normed_digit[i,j]
        count[int(digit)] += 1

    for i in range(10):
        for j in range(num_pixels): mu[i][j] /= count[i]


    #calculate the covariance matrices
    Cov_QDA = [None]*10
    for i in range(10):
        #pick every row of digit_all such that the last column is equal to the current digit class
        Cov_QDA[i] = np.cov(normed_digit[normed_digit[:,-1]==i,:-1].transpose())
    #print "dimenion of normed_digit: ", len(normed_digit[0])
    #print "dimension: ", len(Cov[0]), len(Cov[0][0])
    #print digit_all[digit_all[:,-1]==1,:-1]
    #print "normed_digit: ", normed_digit[normed_digit[:,-1]==3,-1]

    """
    plt.imshow(Cov_QDA[0], cmap='bwr', interpolation='none')
    plt.colorbar()
    plt.show()
    """
    
    #######  LDA  #######
    Cov_LDA = Cov_QDA[0]*count[0]
    for i in range(1,10):
        Cov_LDA += Cov_QDA[i]*count[i]
    Cov_LDA /= size
    #Add a perturbation along the diagonal to make the matrix non-singular
    for i in range(len(Cov_LDA)): Cov_LDA[i,i] += 1e-4
    CovInv = LA.inv(Cov_LDA)

    num_right = 0
    for i in range(10000):
        LDF = -1e20
        x = digit_validate[i,:-1]
        this_digit = -1
        for j in range(10):
            LDF_new = mu[j].dot(CovInv.dot(x)) - 0.5 * mu[j].dot(CovInv.dot(mu[j]))
            if LDF_new > LDF:
                LDF = LDF_new
                this_digit = j
        #print "LDF = ", LDF, "this_digit = ", this_digit
        if this_digit == digit_validate[i,-1]: num_right += 1
        elif this_digit == -1:
            print "Error! No Prediction made!"
            sys.exit()
    accuracy = num_right / 10000.0
    print "For training on ", size, "num_right is ", num_right, " samples, accuracy is ", accuracy
    
    """
    #####  QDA #####
    CovInv_QDA = [None]*10
    for j in range(10):
        for i in range(len(Cov_QDA[j])): Cov_QDA[j][i,i] += 1e-4
    for i in range(10): CovInv_QDA[i] = LA.inv(Cov_QDA[i])
    #Calculate the log determinant as the sum of the log of the eigenvalues
    log_det = [0]*10
    for j in range(10):
        lambda_v, v = LA.eig(CovInv_QDA[j])
        log_lambda = np.log(lambda_v)
        log_det[j] = np.sum(log_lambda)

    num_right = 0
    for i in range(10000):
        QDF = -1e20
        x = digit_validate[i,:-1]
        this_digit = -1
        for j in range(10):
            QDF_new = - (x - mu[j]).transpose().dot(CovInv_QDA[j].dot(x-mu[j]))
            QDF_new -= log_det[j]
            if QDF_new > QDF:
                QDF = QDF_new
                this_digit = j
        if this_digit == digit_validate[i,-1]: num_right += 1
    accuracy = num_right / 10000.0
    print "For training QDA on ", size, "num_right is ", num_right, " samples, accuracy is ", accuracy
"""
"""
    digit_test_raw = scipy.io.loadmat('hw3_mnist_dist/test.mat')
    digit_test = digit_test_raw['testX']
    print "length of digit test: ", len(digit_test)
    print "number of pixels: ", len(digit_test[0])
    print "Id,Category"
    for i in range(len(digit_test)):
        QDF = -1e20
        x = digit_test[i,:]
        this_digit = -1
        for j in range(10):
            QDF_new = - (x - mu[j]).transpose().dot(CovInv_QDA[j].dot(x-mu[j]))
            QDF_new -= log_det[j]
            if QDF_new > QDF:
                QDF = QDF_new
                this_digit = j
        string = str(i) + "," + str(this_digit)
        sys.stdout.write(string)
        print ''
"""









