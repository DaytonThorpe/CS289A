import scipy.io
from sklearn.utils import shuffle
import numpy as np
import sys
import matplotlib.pyplot as plt

def visualize(image):
    image = np.resize(image, (28,28))
    plt.imshow(image)
    plt.show()


def sigmoid(gamma):
    return 1.0 / (1.0 + np.exp(-gamma))

def forward(image, label, V, W):
    vsigmoid = np.vectorize(sigmoid)
    h = np.zeros(len(V))
    z = np.zeros(26)
    h = np.tanh(np.dot(V,image))
    h = np.append(h,1)
    #dropout hidden layer units
    #drop = np.random.randint(2,size=len(h))
    #h = h * drop
    z = vsigmoid(np.dot(W,h))
    loss = 0
    #print "len(label) ", len(label), len(z)
    for i in range(26):
        if label[i] == 1: loss -= np.log(z[i])
        else: loss -= np.log(1-z[i])
    
    return z, h, loss#, drop

def totalLoss(images, labels, V, W):
    totalLoss = 0
    for i in range(len(images)):
        label = np.zeros(26)
        label[labels[i]-1] = 1
        totalLoss += forward(images[i], label, V, W)[2]
    return totalLoss / len(images)

def trainNN(images, labels, num_hidden, epsilon):
    num_images = len(images)

    num_pixels = len(images[0])
    V = np.random.normal(scale=1.0/np.sqrt(num_pixels),size=[num_hidden,num_pixels])
    W = np.random.normal(scale=1.0/np.sqrt(num_hidden + 1),size=[26,num_hidden+1])
    steps = 0
    num_right = 0
    num_wrong = 0
    num_steps = 2 * len(images)
    while (steps < num_steps):
        image = images[steps % num_images]
        #dropout pixels
        #drop = np.random.randint(2,size=len(image))
        #image = image * drop
        label = np.zeros(26)
        label[labels[steps % num_images]-1] = 1
        #forward pass
        #z, h, loss, drop = forward(image, label, num_hidden, V, W)
        z, h, loss = forward(image, label, V, W)
        #print steps, loss
        pred_label = np.argmax(z)
        #backward pass
        grad_w_of_L = np.outer(z - label, np.transpose(h))
        """
        #Checking grading w.r.t. W
        Wup = np.copy(W)
        Wdown = np.copy(W)
        Wup[0,0] += 0.01
        Wdown[0,0] -= 0.01
        print "W's: ", W[0,0], Wup[0,0], Wdown[0,0]
        z_dump, h_dump, LossUp = forward(image, label, num_hidden, V, Wup)
        z_dump, h_dump, LossDown = forward(image, label, num_hidden, V, Wdown)
        grad_w_of_L_FD = (LossUp - LossDown) / 0.02
        print "grad_w_of_L[0,0]: ", grad_w_of_L[0,0]
        print "grad_w_of_L_FD: ", grad_w_of_L_FD
        """
        grad_h_of_L = np.dot(np.transpose(W), np.reshape(z - label, (26,1)))
        grad_h_of_L = grad_h_of_L[:-1] #remove the last row of grad_h_of_L, corresponding to the response of L to changing the constant 1, which we won't do
        grad_v_of_h = np.zeros(V.shape)
        for i in range(num_hidden): grad_v_of_h[i] = (1.0 - h[i]*h[i])*image
        grad_v_of_L = grad_h_of_L * grad_v_of_h
        """
        #Checking gradient w.r.t. V
        Vup = np.copy(V)
        Vdown = np.copy(V)
        Vup[0,10] += 0.01
        Vdown[0,10] -= 0.01
        print "V's: ", V[0,10], Vup[0,10], Vdown[0,10]
        z_dump, h_dump, LossUp = forward(image, label, num_hidden, Vup, W)
        z_dump, h_dump, LossDown = forward(image, label, num_hidden, Vdown, W)
        grad_v_of_L_FD = (LossUp - LossDown) / 0.02
        print "grad_v_of_L[0,10]: ", grad_v_of_L[0,10]
        print "grad_w_of_L_FD: ", grad_v_of_L_FD
        """
        #gradient descent
        W -= epsilon * grad_w_of_L
        V -= epsilon * 2 * grad_v_of_L
        #W -= epsilon * np.dot(grad_w_of_L, np.diag(drop))
        #drop = np.resize(drop, num_hidden)
        #V -= epsilon * 2 * np.dot(np.diag(drop), grad_v_of_L)
        
        if steps % 1000 == 0: print steps, labels[steps % num_images]-1, pred_label
        #if steps % 1000 == 0: print steps, totalLoss(images, labels, V, W)
        if labels[steps % num_images]-1 == pred_label: num_right += 1
        else: num_wrong += 1
        steps += 1
        if steps % num_images == 0: epsilon *= 0.5
    print "accuracy: ", float(num_right) / (num_right + num_wrong)
    return V,W

def predictNN(image, V, W):
    num_hidden = V.shape[0]
    h = np.zeros(num_hidden)
    z = np.zeros(26)
    h = np.tanh(np.dot(V,image))
    h = np.append(h,1)
    h *= 0.5 #dropout
    vsigmoid = np.vectorize(sigmoid)
    z = vsigmoid(np.dot(W,h))
    return np.argmax(z) + 1

#Read in training data
letters = scipy.io.loadmat('hw6_data_dist/letters_data.mat')
#print scipy.io.whosmat('hw6_data_dist/letters_data.mat')
features = letters['train_x']
ones = np.ones([len(features),1])
features = np.append(features,ones,1)
y = letters['train_y']
features_shuffled, y_shuffled = shuffle(features, y)

#visualize(features_shuffled[0])
#sys.exit()

#center and nomralize (except for last column of all ones)
features_normalized = np.ones(features_shuffled.shape)
mean = np.empty(len(features_shuffled[0])-1)
SDev = np.empty(len(features_shuffled[0])-1)
for i in range(len(features_shuffled[0])-1):
    SDev[i] = np.std(features_shuffled[:,i])
    mean[i] = np.mean(features_shuffled[:,i])
    if SDev[i] > 0: features_normalized[:,i] = (features_shuffled[:,i] - mean[i]) / SDev[i]
    else: features_normalized[:,i] = features_shuffled[:,i] - mean[i]

train_size = int(np.floor(0.8*len(y)))

train_features = features_normalized[0:train_size]
train_y = y_shuffled[0:train_size]

val_features = features_normalized[train_size:]
val_y = y_shuffled[train_size:]

learning_rate = 0.003
V, W = trainNN(train_features,train_y, 200, learning_rate)

num_right = num_wrong = 0
for i in range(len(train_features)):
    if train_y[i] == predictNN(train_features[i], V, W): num_right += 1
    else: num_wrong += 1
print "Training accuracy: ", float(num_right) / (num_right + num_wrong)
"""
num_right = num_wrong = 0
for i in range(len(val_features)):
    if val_y[i] == predictNN(val_features[i], V, W): num_right += 1
    else: num_wrong += 1
print "Validation accuracy: ", float(num_right) / (num_right + num_wrong)
"""
num_right = 0
num_wrong = 0
for i in range(len(val_features)):
    prediction = predictNN(val_features[i], V, W)
    if val_y[i] == prediction and num_right < 5:
        print "right! label = ", val_y[i]
        visualize(val_features[i])
        num_right += 1
    elif val_y[i] != prediction and num_wrong < 5:
        print "wrong! label = ", val_y[i], " and prediction was ", prediction
        visualize(val_features[i])
        num_wrong += 1
sys.exit()

#Kaggle Test Data
test_features = letters['test_x']
ones = np.ones([len(test_features),1])
test_features = np.append(test_features,ones,1)
test_normalized = np.ones(test_features.shape)
for i in range(len(mean)):
    if SDev[i] > 0: test_normalized[:,i] = (test_features[:,i] - mean[i]) / SDev[i]
    else: test_normalized[:,i] = test_features[:,i] - mean[i]

print "Id,Category"
for i in range(len(test_features)):
    #print "test image: ", test_normalized[i]
    string = str(i+1) + "," + str(predictNN(test_normalized[i], V, W)) + "\n"
    sys.stdout.write(string)
