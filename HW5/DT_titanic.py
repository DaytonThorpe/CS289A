import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
import csv
import sys
import pandas as pd

class DecisionTree():
    def __init__(self,max_depth,min_size,features,labels):
        self.max_depth = max_depth
        self.min_size = min_size
        self.features = features
        self.labels = labels
        self.correctly_fit = 0
        self.incorrectly_fit = 0

    def entropy(self,set_labels):
        p_c = np.zeros([len(np.unique(self.labels))])
        for i in range(len(set_labels)):
            p_c[set_labels[i]] += 1
        for i in range(len(p_c)): p_c[i] = float(p_c[i]) / len(set_labels)
        entropy = 0
        for i in range(len(p_c)):
            if p_c[i] > 0: entropy -= p_c[i]*np.log2(p_c[i])
        return entropy

    def impurity(self,left_set_labels, right_set_labels):
        num_left = len(left_set_labels)
        num_right = len(right_set_labels)

        entropy_left = 0
        entropy_right = 0
        if num_left > 0: entropy_left = self.entropy(left_set_labels)
        if num_right > 0: entropy_right = self.entropy(right_set_labels)
        
        p_left = float(num_left) / (num_left + num_right)
        p_right = float(num_right) / (num_left + num_right)
        entropy_average = p_left * entropy_left + p_right * entropy_right
        #print "left set labels: ", left_set_labels[:,0]
        #print "left bincount: ", np.bincount(left_set_labels[:,0])
        #print "right set: ", self.labels[right_set,0]
        #print "right bincount: ", np.bincount(right_set_labels[:,0])
        #print "p_left, p_right: ", p_left, p_right
        #print "entropy_left, entropy_right: ", entropy_left, entropy_right
        #print "entropy average: ", entropy_average
        return entropy_average

    def majorityVote(self,set):
        counts = np.bincount(self.labels[set])
        return np.argmax(counts)

    def segmenter(self,set):
        set_labels = self.labels[set]
        #calculate entropy with all elements in one leaf
        entropy = self.entropy(set_labels)
        if entropy == 0:
            return {'feature':-1, 'beta':-1, 'left':set, 'right':-1}
        best_entropy = entropy
        best_feature = -1
        set_features = self.features[set]
        features_transpose = np.transpose(set_features)
        #if entropy < best_entropy:
        best_entropy = entropy
        best_feature = -1
        best_beta = -1
        best_left_set = np.arange(len(set), dtype=int)
        best_right_set = np.zeros([0], dtype=int)
        #loop over features
        for i in range(len(features_transpose)):
            #sort along feature
            sorted_features_index = np.argsort(features_transpose[i]) #returns indices of newly sorted array
            left_set = sorted_features_index
            right_set = np.zeros([0], dtype=int)
            
            #loop over beta's
            for j in range(len(set_labels)):
                #set the cutoff for the feature and move one point from the right histogram to the left histogram
                moving_point = len(sorted_features_index)-1-j
                left_set = sorted_features_index[:moving_point]
                right_set = sorted_features_index[moving_point:]
                if moving_point > 0:
                    if set_features[sorted_features_index[moving_point]][i] == set_features[sorted_features_index[moving_point-1]][i]: continue
                beta = set_features[sorted_features_index[j]][i]
                new_entropy = self.impurity(set_labels[left_set], set_labels[right_set])
                #print new_entropy
                if new_entropy == 0:
                    returned_left_set = np.empty([len(left_set)], dtype = int)
                    for k in range(len(left_set)): returned_left_set[k] = set[left_set[k]]
                    returned_right_set = np.empty([len(right_set)], dtype = int)
                    for k in range(len(right_set)): returned_right_set[k] = set[right_set[k]]
                    return {'feature':i, 'beta':beta, 'left':returned_left_set, 'right':returned_right_set}
                if new_entropy < best_entropy:
                    best_entropy = new_entropy
                    best_feature = i
                    best_beta = beta
                    best_left_set = left_set
                    best_right_set = right_set
        
        #print "best feature: ", best_feature, " best beta: ", best_beta, " best entropy: ", best_entropy
        returned_left_set = np.empty([len(best_left_set)], dtype = int)
        for k in range(len(best_left_set)): returned_left_set[k] = set[best_left_set[k]]
        returned_right_set = np.empty([len(best_right_set)], dtype = int)
        for k in range(len(best_right_set)): returned_right_set[k] = set[best_right_set[k]]
        return {'feature':best_feature, 'beta':best_beta, 'left':returned_left_set, 'right':returned_right_set}

    def update_correctly_fit(self,set,label):
        for i in range(len(set)):
            if self.labels[set[i]] == label: self.correctly_fit += 1
            else: self.incorrectly_fit += 1

    def split(self, node, depth):
        left = node['left']
        right = node['right']
        del(node['left'])
        del(node['right'])

        if depth >= self.max_depth and node['feature'] != -1:
            old_true = self.correctly_fit
            node['left'] = self.majorityVote(left)
            self.update_correctly_fit(left,node['left'])
            #print "trained node A: ", node['left'], "size: ", len(left), "correctly fit new: ", self.correctly_fit - old_true,
            #print "inccorectly fit new: ", len(left) - (self.correctly_fit - old_true)
            old_true = self.correctly_fit
            node['right'] = self.majorityVote(right)
            self.update_correctly_fit(right,node['right'])
            #print "trained node A: ", node['right'], "size: ", len(right), "correctly fit new: ", self.correctly_fit - old_true,
            #print "inccorectly fit new: ", len(right) - (self.correctly_fit - old_true)
            
            return
        if node['feature'] == -1: #there were no more good splits
            node['left'] = node['right'] = self.majorityVote(left)
            old_true = self.correctly_fit
            self.update_correctly_fit(left,node['left'])
            #print "trained node B: ", node['left'], " size: ", len(left), "correctly fit new: ", self.correctly_fit - old_true,
            #print "incorrectly fit new: ", len(left) - (self.correctly_fit - old_true)
            #print "true_label, fit_label: ", self.labels[mergedSet], node
            return

        if len(left) <= self.min_size:
            node['left'] = self.majorityVote(left)
            old_true = self.correctly_fit
            self.update_correctly_fit(left,node['left'])
            #print "trained node C: ", node['left'], " size: ", len(left), "correctly fit new: ", self.correctly_fit - old_true,
            #print "incorrectly fit new: ", len(left) - (self.correctly_fit - old_true)
            #print "true_label, fit_label: ", self.labels[node[2]], left
        else:
            node['left'] = self.segmenter(left)
            self.split(node['left'], depth+1)
        if len(right) <= self.min_size:
            node['right'] = self.majorityVote(right)
            old_true = self.correctly_fit
            self.update_correctly_fit(right,node['right'])
            #print "trained node D: ", node['right'], " size: ", len(right), "correctly fit new: ", self.correctly_fit - old_true,
            #print "incorrectly fit new: ", len(right) - (self.correctly_fit - old_true)
            #print "true_label, fit_label: ", self.labels[node[3]], right
        else:
            node['right'] = self.segmenter(right)
            self.split(node['right'], depth+1)
            
    def train(self):
        wholeSet = np.arange(len(self.labels))
        rootNode = self.segmenter(wholeSet)
        self.split(rootNode, 1)
        return rootNode

    def predict(self, node, sample):
        if type(node) == np.int64: return node
        #node[0] is split feature, node[1] is value
        #print "node: ", node
        if sample[node['feature']] <= node['beta']:
            return self.predict(node['left'], sample)
        else: return self.predict(node['right'], sample)


#read data
df = pd.read_csv('hw5_titanic_dist/titanic_training.csv')
#ignore the column 'education' and treat education-num as a regular continuous variable
numeric_cols = ['pclass', 'age', 'sibsp', 'parch','fare']
x_num = df[numeric_cols].as_matrix()
#cast elements of x_num as floats
for i in range(len(x_num)):
    for j in range(len(x_num[i])):
        x_num[i,j] = float(x_num[i,j])
y = df.survived.as_matrix()
#impute missing values
I = Imputer()
x_num = I.fit_transform(x_num)
cat_cols = ['sex', 'embarked']
cat = df[cat_cols]
x_cat = cat.to_dict(orient='records')

#vectorize
v = DictVectorizer(sparse=False)
v_x_cat = v.fit_transform(x_cat)
print "feature names: ", v.get_feature_names()[4]
I = Imputer(strategy='most_frequent')
v_x_cat = I.fit_transform(v_x_cat)
#combine
x = np.hstack((x_num, v_x_cat))

#shuffle the rows
x_shuffled, y_shuffled = shuffle(x, y)

num_samples = 900#len(x)
train_features = x_shuffled[0:num_samples]
train_labels = y_shuffled[0:num_samples]


test_features = x_shuffled[num_samples:]
test_labels = y_shuffled[num_samples:]

print "max depth, training accuracy, CV accuracy"
for i in range(3,4):
    max_depth = i
    min_size = 3
    classifier = DecisionTree(max_depth, min_size, train_features, train_labels)

    root = classifier.train()
    print root
    print max_depth, float(classifier.correctly_fit) / (classifier.correctly_fit + classifier.incorrectly_fit),

    test_right = test_wrong = 0
    for i in range(len(test_features)):
        #print "train_features, prediction: ", train_features[i], classifier.predict(root, train_features[i])
        if train_labels[i] == classifier.predict(root, train_features[i]): test_right += 1
        else: test_wrong += 1

    print float(test_right) / (test_right + test_wrong)

"""
df_test = pd.read_csv('hw5_titanic_dist/titanic_testing_data.csv')
x_num = df_test[numeric_cols].as_matrix()
cat = df_test[cat_cols]
x_cat = cat.to_dict(orient='records')
v_x_cat = v.fit_transform(x_cat)
test_features = np.hstack((x_num, v_x_cat))

max_depth = 3
min_size = 3
classifier = DecisionTree(max_depth, min_size, train_features, train_labels)

root = classifier.train()

print "Id,Category"
for i in range(len(test_features)):
    string = str(i+1) + "," + str(classifier.predict(root, test_features[i]))
    sys.stdout.write(string)
    print ''
"""







