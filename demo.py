# description of this dataset http://groupware.les.inf.puc-rio.br/har#ixzz2PyRdbAfA
from sklearn import datasets
from sklearn import preprocessing as pp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import numpy as np
import csv
import pandas as pd
import time

# le = pp.LabelEncoder() 
# le.fit(['sitting', 'walking', 'sittingdown', 'standing', 'standingup'])

# initial = time.time()
# ### Retrieving all data
# overall = pd.read_csv("./dataset-har-PUC-Rio-ugulino.csv", delimiter=';', header='infer') 
# data = overall.loc[:, "x1":"z4"].as_matrix() # has to be converted to ndarray in order to be processed by segment_signal()
# targets = overall.loc[:,"class,,"].as_matrix() # double commas: looks like the researchers are naughty

# load = time.time()
# print "--- time to load and select datasets: %s seconds ---" % (load - initial)

### Data segmentation: shall use a sudden change of sensor readings
### like if (x_pre - x_curr <= 1.0, do nothing)
### Range of Accelerometer sensor readings is +3g/-3g

# reading 14 sets of data in every 2 seconds. 
# For segmenting the data from online only. 
# each set of data is taken 150ms apart from another.
# so choosing a window size of 14 will be 2.1 seconds.


# def segment_signal(data, window_size=14): 

#     N = data.shape[0]
#     dim = data.shape[1]
#     K = N/window_size
#     segments = numpy.empty((K, window_size, dim))
#     for i in range(K):
#         segment = data[i*window_size:i*window_size+window_size,:]
#         segments[i] = numpy.vstack(segment)
#     return segments



# ##!!!! questions: for normalization, should it be done right after loading csv or after segmenation? 
# ##!!!! Normalize() can't process nadarray with dimension > 2.
# X = pp.normalize(data)
# y = targets[::14] 
# # y = y[:-1]# -1 because it will have a extra set of data than X.

# normalizing = time.time()
# print "--- time to normalize: %s seconds ---" % (normalizing - load)

# segs = segment_signal(X)

# segmenting = time.time()
# print "--- time to segment: %s seconds ---" % (segmenting - normalizing)

### feautre extraction // take the difference between sensors

### this method is to extract the difference between consecutive sensor readings.
## parameter raw is a 2D ndarray
## return a 2D ndarray
def extract_diff(raw):

    N = raw.shape[0] # number of sets of sensor readings
    dim = raw.shape[1] # number of values in each readings
    features = numpy.empty((N - 1, dim))
    for i in range(1, N):
        for j in range(dim):
            features[i-1][j] = raw[i][j] - raw[i-1][j]

    return features

def extract_diff_2(raw):

    N = raw.shape[0] # number of segments of sensor readings ()
    I = raw.shape[1] # number of sets of readings (14)
    J = raw.shape[2] # number of values in each set of readings (12)
    feature_num = (I - 1) * J
    feature = numpy.empty((feature_num))
    features = numpy.empty((N, feature_num))
    for n in range(N):
        idx = 0;
        for i in range(1, I):
            for j in range(J):
                feature[idx] = raw[n][i][j] - raw[n][i-1][j]
                idx += 1
        features[n] = feature
        

    return features

# # features = extract_diff_2(segs)

# extracting_feature = time.time()
# print "--- time to extract features: %s seconds ---" % (extracting_feature - segmenting)


#################################################################################################################################
#### generate data
##################################################################################################################################

import random
def gen_wavehands():
    data = []
    for j in range(156):
        data.append(random.uniform(-6,-4))
    return data, "walking"

def gen_busdriver():
    data = []
    for j in range(156):
        data.append(random.uniform(-4,-2))
    return data, "busdriver"

def gen_frontback():
    data = []
    for j in range(156):
        data.append(random.uniform(-2,0))
    return data, "fontback"

def gen_sidestep():
    data = []
    for j in range(156):
        data.append(random.uniform(0,2))
    return data, "sidestep"

def gen_jumping():
    data = []
    for j in range(156):
        data.append(random.uniform(2,4))
    return data, "jumping"

def gen_jumpingjack():
    data = []
    for j in range(156):
        data.append(random.uniform(4,6))
    return data, "jumpingjack"

def gen_turnclap():
    data = []
    for j in range(156):
        data.append(random.uniform(-8,-6))
    return data, "turnclap"

def gen_squatturnclap():
    data = []
    for j in range(156):
        data.append(random.uniform(6,8))
    return data, "squatturnclap"

def gen_window():
    data = []
    for j in range(156):
        data.append(random.uniform(-10,-8))
    return data, "window"

def gen_window360():
    data = []
    for j in range(156):
        data.append(random.uniform(8,10))
    return data, "window360"

data_list = []
label_list = []

from random import randint

funct_list = [gen_wavehands, gen_busdriver, gen_frontback, gen_sidestep, gen_jumping,
              gen_jumpingjack]#, gen_window] #gen_window360] gen_turnclap gen_squatturnclap]#
for i in range(150000):
    rand_funct = randint(0, len(funct_list)-1)
    data, label = funct_list[rand_funct]()
    data_list.append(data)
    label_list.append(label)
#     data_list.append(data)
#     label_list.append(label)
# for i in range (500):
#     data, label = gen_squatturnclap()
#     data_list.append(data)
#     label_list.append(label)
# for i in range(500):
#     data, label = gen_window()
#     data_list.append(data)
#     label_list.append(label)
# for i in range (500):
#     data, label = gen_window360()
#     data_list.append(data)
#     label_list.append(label)
    
#     [data_list, label_list]
# print type(data_list).shape
data_list = np.asarray(data_list)
label_list = np.asarray(label_list)

# np.asarray(label_list).shape

############################################################################################################################

le = pp.LabelEncoder() 
le.fit(['wavehands', 'busdriver', 'frontback', 'sidestep', 'jumping',
        'jumpingjack', 'turnclap', 'squatturnclap', 'window', 'window360'])

X = pp.normalize(data_list)
y = label_list


layer_1_val = 15
layer_2_val = 10
################################################################
####having 15 neurons
kfold = KFold(n_splits=2, shuffle=True)
fold_index = 0
for train, test in kfold.split(X):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(layer_1_val, layer_2_val), random_state=1).fit(X[train], y[train])
    predictions = clf.predict(X[test])
    accuracy = clf.score(X[test], y[test])
    cm = confusion_matrix(y[test], predictions)

    print('In the %i fold, the classification accuracy is %f' %(fold_index, accuracy))
    print('And the confusion matrix is: ')
    print(cm)
    fold_index += 1
#############################################################
##### Choose by uncommenting on either one
#############################################################
# kfold = KFold(n_splits=10, shuffle=True)

# fold_index = 0
# for train, test in kfold.split(features):
#     svm = SVC(kernel = 'linear', C = 50).fit(features[train], y[train])
#     svm_predictions = svm.predict(features[test])
#     recall = recall_score(y[test], svm_predictions, average='macro') # 
#     accuracy = svm.score(features[test], y[test])
#     cm = confusion_matrix(y[test], svm_predictions)

#     print('In the %i fold, the classification accuracy is %f and the recall is %f' %(fold_index, accuracy, recall))
#     print('And the confusion matrix is: ')
#     print(cm)
#     fold_index += 1
################################################################


# evaluate_model = time.time()
# print "--- time to extract features: %s seconds ---" % (evaluate_model - extracting_feature)



