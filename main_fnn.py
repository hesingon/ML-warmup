# description of this dataset http://groupware.les.inf.puc-rio.br/har#ixzz2PyRdbAfA
from sklearn import datasets
from sklearn import preprocessing as pp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import numpy
import csv
import pandas as pd
import time

le = pp.LabelEncoder() 
le.fit(['sitting', 'walking', 'sittingdown', 'standing', 'standingup'])

initial = time.time()
### Retrieving all data
overall = pd.read_csv("./dataset-har-PUC-Rio-ugulino.csv", delimiter=';', header='infer') 
data = overall.loc[:, "x1":"z4"].as_matrix() # has to be converted to ndarray in order to be processed by segment_signal()
targets = overall.loc[:,"class,,"].as_matrix() # double commas: looks like the researchers are naughty

load = time.time()
print "--- time to load and select datasets: %s seconds ---" % (load - initial)


### Data segmentation: shall use a sudden change of sensor readings
### like if (x_pre - x_curr <= 1.0, do nothing)
### Range of Accelerometer sensor readings is +3g/-3g

# reading 14 sets of data in every 2 seconds. 
# For segmenting the data from online only. 
# each set of data is taken 150ms apart from another.
# so choosing a window size of 14 will be 2.1 seconds.


def segment_signal(data, window_size=14): 

    N = data.shape[0]
    dim = data.shape[1]
    K = N/window_size
    segments = numpy.empty((K, window_size, dim))
    for i in range(K):
        segment = data[i*window_size:i*window_size+window_size,:]
        segments[i] = numpy.vstack(segment)
    return segments



##!!!! questions: for normalization, should it be done right after loading csv or after segmenation? 
##!!!! Normalize() can't process nadarray with dimension > 2.
X = pp.normalize(data)
y = targets[::14] 
y = y[:-1]# -1 because it will have a extra set of data than X.

normalizing = time.time()
print "--- time to normalize: %s seconds ---" % (normalizing - load)

segs = segment_signal(X)

segmenting = time.time()
print "--- time to segment: %s seconds ---" % (segmenting - normalizing)

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

features = extract_diff_2(segs)

extracting_feature = time.time()
print "--- time to extract features: %s seconds ---" % (extracting_feature - segmenting)

layer_1_val = 15
layer_2_val = 10

#having 15 neurons
kfold = KFold(n_splits=10, shuffle=True)
fold_index = 0
for train, test in kfold.split(features):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(layer_1_val, layer_2_val), random_state=1).fit(features[train], y[train])
    predictions = clf.predict(features[test])
    accuracy = clf.score(features[test], y[test])
    cm = confusion_matrix(y[test], predictions)

    print('In the %i fold, the classification accuracy is %f' %(fold_index, accuracy))
    print('And the confusion matrix is: ')
    print(cm)
    fold_index += 1


evaluate_model = time.time()
print "--- time to extract features: %s seconds ---" % (evaluate_model - extracting_feature)



