import numpy as np
import csv
import pandas as pd
import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as pp

import cPickle as pickle

WAVEHANDS = 0
BUSDRIVER = 1
FRONTBACK = 2
SIDESTEP = 3
TURNCLAP = 4
SQUAT = 5
WINDOW = 6
WINDOW360 = 7
JUMP = 8
JUMPINGJACK = 9


# input directory and filename of the dataset
sensorNum = 3
sensorAxis = 6
window_size = 40  # 40 * 50ms = 2000ms
sample_rate = 1  # determines the time interval to sample. E.g. 2 means take sample every 100ms instead of 50


def fast_convert(path):
    xl = pd.read_csv(path, header=None)

    return xl.iloc[:500].as_matrix(), xl.iloc[500:].as_matrix()




    # # training_data = np.empty((0, sensorNum * sensorAxis * window_size))
    # # training_label = np.empty(0)
    # training_data = []
    # training_label = []
    #
    #     dance_set = []
    #
    #     for j in range(len(sheet)/(3*sample_rate)):
    #         # Read one sheet
    #         reading = sheet.iloc[3*j*sample_rate:3*j*sample_rate+3, 1:7].as_matrix().flatten()  # store every 3 rows as one reading
    #         # dance_set = np.vstack((dance_set, reading))                 # store readings together
    #         dance_set.append(reading)
    #
    #     for j in range(len(dance_set) - window_size + 1):
    #         # combine several reading into each window
    #         # training_data = np.vstack((training_data, np.hstack(dance_set[j:j+window_size])))
    #         # training_label = np.hstack((training_label, i))
    #         training_data.append(np.hstack((dance_set[j:j+window_size])))
    #         training_label.append(i)
    #
    #
    # return training_data, training_label

train_id = []
test_id = []

X_train = []
X_test = []

for i in range(11):
    train, test = fast_convert('yp/dance_data/{}.csv'.format(i))
    if len(train_id) <= 0:
        train_id.append(len(train))
        test_id.append(len(test))
    else:
        train_id.append(len(train) + train_id[-1])
        test_id.append(len(test) + test_id[-1])
    X_train += train.tolist()
    X_test += test.tolist()

X_train = np.array(X_train).astype(np.float64)
scaler = pp.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

train_input = []
train_label = []
for i in range(11):
    if i == 0:
        data_id_range = range(0, train_id[i] - window_size)
    else:
        data_id_range = range(train_id[i-1], train_id[i] - window_size)

    for j in data_id_range:
        train_input.append(np.hstack(X_train[j:j + window_size]))
        train_label.append(i)


X_test = scaler.transform(X_test)
test_input = []
test_label = []
for i in range(11):
    if i == 0:
        data_id_range = range(0, test_id[i] - window_size)
    else:
        data_id_range = range(test_id[i-1], test_id[i] - window_size)

    for j in data_id_range:
        test_input.append(np.hstack(X_test[j:j + window_size]))
        test_label.append(i)

print np.array(test_input).shape
print np.array(test_label).shape


layer_1_val = 40
layer_2_val = 20
layer_3_val = 25

# Fit into model
clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=500, hidden_layer_sizes=(layer_1_val, layer_2_val, layer_3_val),
                    random_state=1).fit(train_input, train_label)


pickle.dump( clf, open("model40.pkl", "wb"))


# Preprocess sample 2
# testing = pp.normalize(np.array(training3))
# test_label = np.array(label3)
from sklearn.utils import shuffle
test_input, test_label = shuffle(test_input, test_label, random_state=0)

# Test model with another sample
predictions = clf.predict(test_input)
accuracy = clf.score(test_input, test_label)
cm = confusion_matrix(test_label, predictions)

print('The classification accuracy is %f' %(accuracy))
print('And the confusion matrix is: ')
print(cm)

