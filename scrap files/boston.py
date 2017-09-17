from sklearn import datasets

bst = datasets.load_boston()
iris = datasets.load_iris()

#print bst.DESCR
print iris.target.shape
print iris.data.shape
print iris.target
#print bst.target 