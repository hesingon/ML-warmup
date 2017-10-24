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
def fast_convert(path):
    sensorNum = 3
    sensorAxis = 6
    window_size = 40    # 40 * 50ms = 2000ms
    sample_rate = 1     # determines the time interval to sample. E.g. 2 means take sample every 100ms instead of 50

    xl = pd.ExcelFile(path) # find the number of spreadsheets in the file
    num_sheet = len(xl.sheet_names)

    # training_data = np.empty((0, sensorNum * sensorAxis * window_size))
    # training_label = np.empty(0)
    training_data = []
    training_label = []

    for i in range(num_sheet):
        sheet = xl.parse(i, header=None)
        print len(sheet)
        # dance_set = np.empty((0, sensorNum * sensorAxis))
        dance_set = []

        for j in range(len(sheet)/(3*sample_rate)):
            # Read one sheet
            reading = sheet.iloc[3*j*sample_rate:3*j*sample_rate+3, 1:7].as_matrix().flatten()  # store every 3 rows as one reading
            # dance_set = np.vstack((dance_set, reading))                 # store readings together
            dance_set.append(reading)

        for j in range(len(dance_set) - window_size + 1):
            # combine several reading into each window
            # training_data = np.vstack((training_data, np.hstack(dance_set[j:j+window_size])))
            # training_label = np.hstack((training_label, i))
            training_data.append(np.hstack((dance_set[j:j+window_size])))
            training_label.append(i)


    return training_data, training_label

training1, label1 = fast_convert('./training1.xlsx')
training2, label2 = fast_convert('./training2.xlsx')
training3, label3 = fast_convert('./training3.xlsx')
# training4, label4 = fast_convert('./training4.xlsx')

# training = pp.normalize(np.array(training1 + training2))
# label = np.array(label1+label2)

# layer_1_val = 40
# layer_2_val = 20
# layer_3_val = 25
#
# kfold = KFold(n_splits=3, random_state=2, shuffle=True)
# fold_index = 0
# for train_id, test_id in kfold.split(training):
#     clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=200,
#                      hidden_layer_sizes=(layer_1_val, layer_2_val, layer_3_val), random_state=1).fit(training[train_id], label[train_id])
#     predictions = clf.predict(training[test_id])
#     accuracy = clf.score(training[test_id], label[test_id])
#     cm = confusion_matrix(label[test_id], predictions)
#
#     print('In the %i fold, the classification accuracy is %f' %(fold_index, accuracy))
#     print('And the confusion matrix is: ')
#     print(cm)
#     fold_index += 1


training = pp.normalize(np.array(training1 + training2))
label = np.array(label1 + label2)

layer_1_val = 40
layer_2_val = 20
layer_3_val = 25

# Fit into model
clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=500,
                 hidden_layer_sizes=(layer_1_val, layer_2_val, layer_3_val), random_state=1).fit(training, label)


pickle.dump( clf, open("model40.pkl", "wb"))


# Preprocess sample 2
testing = pp.normalize(np.array(training3))
test_label = np.array(label3)
from sklearn.utils import shuffle
testing, test_label = shuffle(testing, test_label, random_state=0)

# Test model with another sample
predictions = clf.predict(testing)
accuracy = clf.score(testing, test_label)
cm = confusion_matrix(test_label, predictions)

print('The classification accuracy is %f' %(accuracy))
print('And the confusion matrix is: ')
print(cm)

