import scipy.io
import numpy as np
from sklearn.utils import shuffle

class DecisionTree(object):
    def split(split_index, sorted_indices):
        S_left = sorted_indices[:split_index]
        S_right = sorted_indices[split_index:]
        return S_left, S_right

    def train(self, features, labels, depth, max_depth):
        #nodes = np.array(empty)
        #loop over features
        best_entropy = 1
        best_feature = -1
        features_transpose = np.transpose(features)
        for i in range(len(features[0])):
            #sort along feature
            sorted_features_index = np.argsort(features_transpose[i]) #returns indices of newly sorted array
            #calculate entropy for every single element being in the right leaf
            left_count = 0
            right_count = len(features)
            p_spam_left = 0
            p_spam_right = float(np.sum(labels)) / len(labels)
            entropy = -(p_spam_right * np.log2(p_spam_right) + (1 - p_spam_right) * np.log2(1 - p_spam_right))
            #if the entropy is 0, no need to split
            
            #print "initial entropy ", entropy
            beta = -1
            beta_index = -1
            for j in range(len(features)):
                #update the sizes and spam probabilities in both leaves
                p_spam_left = float(p_spam_left * left_count + labels[sorted_features_index[j]]) / (left_count + 1)
                if right_count > 1: p_spam_right = (p_spam_right * right_count - labels[sorted_features_index[j]]) / (right_count - 1)
                else: p_spam_right = 0
                left_count += 1
                right_count -= 1
                
                p_left = float(left_count) / (left_count + right_count)
                p_right = float(right_count) / (left_count + right_count)

                entropy_left = 0
                entropy_right = 0
                
                if p_spam_left > 0 and p_spam_left < 1.0:
                    entropy_left = -(p_spam_left * np.log2(p_spam_left) + (1 - p_spam_left) * np.log2(1 - p_spam_left))
                if p_spam_right > 0 and p_spam_right < 1.0:
                    entropy_right = -(p_spam_right * np.log2(p_spam_right) + (1 - p_spam_right) * np.log2(1 - p_spam_right))
                new_entropy = p_left * entropy_left + p_right * entropy_right
                #print p_spam_left, p_spam_right, new_entropy
                if new_entropy < entropy:
                    entropy = new_entropy
                    beta = features[j][i]
                    beta_index = j
            print "best beta for feature ", i, " is ", beta, " with entropy ", entropy
            if entropy < best_entropy:
                best_entropy = entropy
                best_feature = i
        print "best feature is ", best_feature, " with beta ", beta, " gives entropy ", best_entropy
    

        return 0


#Read in training data
spam = scipy.io.loadmat('dist/spam_data.mat')
spam_features = spam['training_data'] #23702 samples with 32 features
spam_labels = spam['training_labels'].transpose()

#shuffle the rows
features_shuffled, labels_shuffled = shuffle(spam_features, spam_labels)

train_features = features_shuffled[0:1000]
train_labels = labels_shuffled[0:1000]

classifier = DecisionTree()

max_depth = 10
classifier.train(train_features, train_labels, 0, max_depth)




