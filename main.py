# description of this dataset http://groupware.les.inf.puc-rio.br/har
from sklearn import datasets
from sklearn import preprocessing as pp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy
import csv
import pandas as pd

# ### NOTES: RANGE OF VALUES CORRESPONDING TO TARGET 
#     2 ~ 50632: 0; 
#     50633 ~ 62459: 2; 
#     62460 ~ 109829: 3; 
#     109830 ~ 122243: 4; 
#     122244 ~ 165633: 1;


#About labelEncoder
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.fit
le = pp.LabelEncoder() 
le.fit(['sitting', 'walking', 'sittingdown', 'standing', 'standingup'])

data = numpy.genfromtxt('subset.csv', delimiter=";", dtype="string") # for the original sample, should skip 1 line
targets = numpy.genfromtxt('subset_target.csv')

## Display the data to verify the data and targets have the correct number of dimensions.

# print type(data)
# print data.shape
# print data

# print type(targets)
# print targets.shape
# print targets
###############################################

## Step 0, Data Segmentation
## ~~~~ Need to modify segmentation

def segment_signal(data, window_size=2): #each set of data is taken 150ms apart from another.

    N = data.shape[0]
    dim = data.shape[1]
    K = N/window_size
    segments = numpy.empty((K, window_size, dim))
    for i in range(K):
        segment = data[i*window_size:i*window_size+window_size,:]
        segments[i] = numpy.vstack(segment)
    return segments

segs = segment_signal(data)

# #test
# print type(segs)
# print segs.shape
# print segs
####################################################


## Step 1, Data-Preprosessing

X = data
y = targets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
# taken from sklearn's confusion matrix example

# #test
# print type(X_train)
# print X_train.shape
# print X_train

# print type(X_test)
# print X_test.shape
# print X_test

# print type(y_train)
# print y_train.shape
# print y_train

# print type(X_test)
# print X_test.shape
# print X_test

normalized_X = pp.normalize(X)


####################################################


## Step 2, Model Selection


# svm = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
# svm_predictions = svm.predict(X_test)  
# ## ! Test result is now stored in svm_predictions

# #test
# print svm_predictions

####################################################



## Step 3: Evaluated the model

kfold = KFold(n_splits=10, shuffle=True)

fold_index = 0
for train, test in kfold.split(normalized_X):
    svm = SVC(kernel = 'linear', C = 1).fit(normalized_X[train], y[train])
    svm_predictions = svm.predict(normalized_X[test])
    accuracy = svm.score(normalized_X[test], y[test])
    cm = confusion_matrix(y[test], svm_predictions)

    print('In the %i fold, the classification accuracy is %f' %(fold_index, accuracy))
    print('And the confusion matrix is: ')
    print(cm)
    fold_index += 1

####################################################


