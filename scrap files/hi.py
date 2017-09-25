# description of this dataset http://groupware.les.inf.puc-rio.br/har
from sklearn import datasets
from sklearn import preprocessing as pp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import numpy
import csv
import pandas as pd

le = pp.LabelEncoder() 
le.fit(['sitting', 'walking', 'sittingdown', 'standing', 'standingup'])

data = numpy.genfromtxt('subset.csv', delimiter=";", dtype="string") # for the original sample, should skip 1 line
targets = numpy.genfromtxt('subset_target.csv')