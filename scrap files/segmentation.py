# description of this dataset http://groupware.les.inf.puc-rio.br/har
from sklearn import datasets
import numpy as np
import csv

iris = datasets.load_iris()

# def segment_signal(data, window_size=2):

#     N = data.shape[0]
#     dim = data.shape[1]
#     K = K/window_size
#     segments = numpy.empty((K, window_size, dim))
#     for i in range(K):
#         segment = data[i*window_size:i*window_size+window_size,:]
#         segments[i] = numpy.vstack(segment)
#     return segments

data = np.genfromtxt('sample.csv', delimiter=";")
data2 = np.genfromtxt('sample.csv', delimiter=";", dtype="string")
data3 = np.genfromtxt('sample.csv', delimiter=";", dtype="string", skip_header=1)
print type(data)
print data.shape
print data
print data2
print data3
