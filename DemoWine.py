
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from LaPOLeaFr import LaPOLeaF
import MyLeadingTree as lt
import numpy as np
from collections import Counter

import time

def LabeledInds(Labels):
    return  [i for i in range(np.shape(Labels)[0]) if sum(Labels[i,:])>0]

lt_num =8  # 子树个数
wine = datasets.load_wine()
X = wine.data
y = wine.target

# lt_num = 10  # 子树个数
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target


ND = len(y)
classNum = len(np.unique(y))
labeledNumPerClass = 2
labeledCnt = classNum * labeledNumPerClass
t1= time.time()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
lt1 = lt.LeadingTree(X_train=X, dc=0.2, lt_num=lt_num)  # 整个数据集构造引领树
lt1.fit()



LaPOLeaFojb = LaPOLeaF()
Centers = lt1.Q[:lt_num]
eachAL = LaPOLeaFojb.ConstructOLeaF(lt1.Pa,Centers)
InitLables = LaPOLeaFojb.InitLabels(y,classNum,labeledNumPerClass)
c2pLabels = LaPOLeaFojb.c2pPropagation(eachAL,lt1.layer,InitLables,lt1.delta)


DistCentersR = lt1.D[Centers]
DistCenters = DistCentersR[:, Centers]
labelR2R = LaPOLeaFojb.r2rPropagation(c2pLabels, DistCenters, lt1.density, Centers)
#print(LabeledInds(labelR2R))


finalLabels = LaPOLeaFojb.p2cPropagation(eachAL, lt1.layer, labelR2R, lt1.delta)
#print(LabeledInds(finalLabels))

predictY = [ np.argmax(finalLabels[i,:]) for i in range(ND)]
arr = y-predictY
count = Counter(arr)[0]
print("总准确率为", (count-labeledCnt) / (ND - labeledCnt))

t2= time.time()
print("Time elapse: ", t2-t1)