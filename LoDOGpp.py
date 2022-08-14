###This is to remove the parameters from LoDOG
####Alanxu@IMI,GUES 2020-3-30

import pandas as pd
from datetime import datetime
import math
import time
import cv2
import matplotlib.pyplot as plt
from numpy import mat
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from LaPOLeaFr import LaPOLeaF

# def EuclidianDist(MatP,MatQ):
#     m, d = MatP.shape[0], MatP.shape[1]
#     n = MatQ.shape[0]
#     Pone = np.ones((d, n))
#     Qone = np.ones((m, d))
#     pSqr = np.power(MatP, 2)
# 
#     T1 = np.matmul(pSqr, Pone)
#     qT = np.transpose(MatQ)
#     qSqr = np.power(qT , 2)
#     T2 = np.matmul(Qone, qSqr)
#     res = T1 + T2 -2 * np.matmul(MatP, qT)
#     res[res<0]=0 ##Remove calculation error
#     return np.sqrt(res)

### localDensity2 is used to verify the python version localDensity(D,dc)
# def localDensity2(D,dc):
#     ND = np.shape(D)[0]
#     rho = np.zeros((ND, 1))
#     for i in range(0, ND-1):
#         for j in range(i+1, ND):
#             rho[i] = rho[i] + np.exp(-(D[i, j] / dc) * (D[i, j] / dc))
#             rho[j] = rho[j] + np.exp(-(D[i, j] / dc) * (D[i, j] / dc))
#     return rho

def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM= np.sum(X1 ** 2, 1).reshape(-1, 1) ##行数不知道，只知道列数为1
    tempN= np.sum(X2 ** 2, 1)# X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return  np.sqrt(sqdist)

def DistNormalize(D):
    ### Normalize the Dist martix to eliminate the function H()
    ### Normalize Dist Matrix is not necessary, maybe normalize delta is OK!
    ColSum = np.array(np.sum(D, 0)) ## (D, 1) is wrong!
    return (D/sum(ColSum)*np.shape(D)[0])

def readData(csvFileName):
    df = pd.read_csv(csvFileName)  ## ignore the head line of the file.
    return df.values

def localDensity(D,dc):
    #tempMat = np.exp(-((D/dc) ** 2))
    tempMat1 = np.exp(-(D ** 2))
    tempMat = np.power(tempMat1, dc ** (-2))
    return np.sum(tempMat, 1)-1 ### exclude the self-contribution, and the function is validated!

def baseMat(D):
    tempMat1 = np.exp(-(D ** 2))
    return tempMat1

def ObjValue(deltaVec, Ng, ordGamDesc, alpha):
    centDelta=deltaVec[ordGamDesc[0: Ng]]
    return alpha*Ng + (1-alpha)*(sum(deltaVec)-sum(centDelta))


def DeltaZeta(D,Rho):
    # Compute the delta distance and leading node index
   sortedInd = np.argsort(Rho)
   descInd = sortedInd[::-1] ## reverse the array
   ND = len(Rho)
   delta = np.zeros(ND)
   zeta = np.ones(ND) * (-1)
   dvec = D[descInd[0], :]
   delta[descInd[0]] = max(dvec)## the delta distance for the point of greatest rho
   for i in range(1, ND):
       curInd=descInd[i]
       greaterInds = descInd[0:i]
       distVec= D[curInd, greaterInds]## retrieve the distances with greater rho
       delta[curInd] = min(distVec)
       zeta[curInd] = greaterInds[np.argmin(distVec)]
   return delta, zeta

def zeta2edges(zeta):
    edgesO = np.array(list(zip(range(len(zeta)), zeta)))
    ind = edgesO[:, 1] > -1
    edges = edgesO[ind,]
    return edges

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 23,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}


def LabeledInds(Labels):
    return  [i for i in range(np.shape(Labels)[0]) if sum(Labels[i,:])>0]


def drawLT2D(dataSetR, zeta, colorStr, i):
    #### Only for DNAdataset
    #idsStr = "a,b,c,d,e,f,g,h,i,j"
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }

    plt.subplot(1, 3, 1 + i)
    titles = ['a', 'b', 'c']
    idsStr = "0,1,2,3,4,5,6,7,8,9"
    IDsArr = idsStr.split(',')
    G = nx.MultiDiGraph()
    edges = zeta2edges(zeta)
    G.add_edges_from(edges)
    pos = {ind: dataSetR[ind, :] for ind in range(ND)}
    nlabels = dict(zip(range(ND), IDsArr))
    nx.draw_networkx(G, pos, node_color=colorStr, labels=nlabels, font_color="#FFFF00")
    if i > 0:
        #plt.xticks([])
        plt.yticks([])
    plt.xlabel("(" + titles[i] + ")", font1)
    plt.tick_params(labelsize=14)

    #plt.show()

def ColorStrArray(rho):
    rhoNormal= (rho-min(rho))/(max(rho)-min(rho))
    ND = len(rho)
    #colorArray =rhoNormal * 0xFFFFFF
    colorArrayRed  = (0xFF)*rhoNormal
    colorArrayBlue = 0xFF - (0xFF) * rhoNormal
    colorStrRed = ['#{:02X}'.format(math.floor(e)).upper() for e in colorArrayRed]
    colorStrBlue = ['00{:02X}'.format(math.floor(e)).upper() for e in colorArrayBlue]
    colorStr = [colorStrRed[i] + colorStrBlue[i] for i in range(ND)]
    # = [colorE.toString(16).toUpperCase() for colorE in colorArInt]
    return colorStr

dataSetR = readData('./data/DNAdataPoints.csv')
AllLabels = readData('./data/DNAdataLabels.csv')
GivenLabels = LaPOLeaF.InitLabels(AllLabels, 2, 2)
# dataSet = readData('./data/Wine.csv')
# dataSetR1 = dataSet[:, 1:14]
# mms = MinMaxScaler()
# dataSetR = mms.fit_transform(dataSetR1)
# AllLabels = dataSet[:, 14] - 1
GivenLabels = LaPOLeaF.InitLabels(AllLabels, 2, 2)

ND = np.shape(dataSetR)[0]
D2 = EuclidianDist2(dataSetR, dataSetR)

###Search zeta mutation 2020-3-31



###For paper presentation


rho = localDensity(D2, 0.5)
delta, zeta = DeltaZeta(D2, rho)
colorStr2 = ColorStrArray(rho)
drawLT2D(dataSetR, zeta, colorStr2, 0)


rho2 = localDensity(D2, 2)
delta, zeta = DeltaZeta(D2, rho2)
colorStr3 = ColorStrArray(rho2)
drawLT2D(dataSetR, zeta, colorStr3, 1)

rho3 = localDensity(D2, 2)
delta, zeta = DeltaZeta(D2, rho3)
colorStr = ColorStrArray(rho3)
drawLT2D(dataSetR, zeta, colorStr, 2)

plt.show()



gamma = rho * delta
gammaSortInd = np.argsort(gamma)
GamIndDesc = gammaSortInd[::-1] ## omit start, end the set the step '-1'

#deltaN2 = deltaN/sum(deltaN)*ND
deltaN2 = delta/sum(delta)*ND ## normalize delta!!!!
PossibleNg = min(9, ND)
objVal = np.zeros(PossibleNg)
objVal2 = np.zeros(PossibleNg)
# objVal3 = np.zeros(PossibleNg)
for Ng in range(1, PossibleNg+1):
    objVal[Ng-1] = ObjValue(deltaN2, Ng, GamIndDesc, 0.55)
    objVal2[Ng - 1] = ObjValue(deltaN2, Ng, GamIndDesc, 0.5)
    # objVal3[Ng - 1] = ObjValue(deltaN2, Ng, GamIndDesc, 0.6)
    # objVal[Ng - 1] = ObjValue2(deltaN2, Ng, GamIndDesc)
plt.plot(range(1, PossibleNg+1), objVal, '--', label= r'$\alpha=0.55$')  ###comment for debugging

plt.plot(range(1, PossibleNg+1), objVal2,  label= r'$\alpha=0.5$')
# # plt.plot(range(1, PossibleNg+1), objVal3)

plt.legend()
plt.show()

# Ng = 2 ###decide dynamically
# Centers = GamIndDesc[:Ng]
#
# AL = LaPOLeaF.ConstructOLeaF(zeta.astype(np.int), Centers)
# AllLayerInds = {}
# for i in range(Ng): ###find the layerInd for each sub-LT
#     layerInds = LaPOLeaF.LayerInd(AL, GamIndDesc[i])
#     AllLayerInds.update(layerInds)
#
# LayerIndsSort = sorted(AllLayerInds.items(), key = lambda AllLayerInds:AllLayerInds[0])
# LayerIndsArray = [el[1] for el in LayerIndsSort]
# #### demonstrate the sensitivity of dc.
#
#
# print(LabeledInds(GivenLabels))
# labelC2P = LaPOLeaF.c2pPropagation(AL, LayerIndsArray, GivenLabels, delta)
# print(LabeledInds(labelC2P))
#
# DistCentersR = D2[Centers]
# DistCenters = DistCentersR[:, Centers]
# labelR2R = LaPOLeaF.r2rPropagation(labelC2P, DistCenters, rho, Centers)
# print(LabeledInds(labelR2R))
#
#
# finalLabels = LaPOLeaF.p2cPropagation(AL, LayerIndsArray, labelR2R, delta)
# print(LabeledInds(finalLabels))
# print(finalLabels)

dcs = np.linspace(0.2, 9, 100)
dcNum = len(dcs)

OptObjValues = np.zeros(dcNum)
optNg = np.zeros(dcNum)
rho2Vec = np.zeros(dcNum)
rho3Vec = np.zeros(dcNum)

i = 0
for dc in dcs:
    rho2 = localDensity(D2, dc)
    delta2, zeta2 = DeltaZeta(D2, rho2)
    gamma2 = rho2 * delta2
    gammaSortInd = np.argsort(gamma2)
    GamIndDesc = gammaSortInd[::-1]
    deltaN2 = delta / sum(delta) * ND  ##normalize delta
    objVal = np.zeros(PossibleNg)
    objValOrig = np.zeros(PossibleNg)
    for Ng in range(1, PossibleNg + 1):
        objVal[Ng - 1] = ObjValue(deltaN2, Ng, GamIndDesc, 0.5)
        objValOrig[Ng - 1] = ObjValue(delta2, Ng, GamIndDesc, 0.5)
    optNg[i] = np.argmin(objVal)+1  ###very important +1!
    OptObjValues[i] = np.min(objVal)
    #OptObjValues[i] = np.min(objValOrig)
    rho2Vec[i] = rho2[2]
    rho3Vec[i] = rho2[3]
    i += 1


plt.plot(dcs, optNg, '--', label="Optimal Ng")

plt.plot(dcs, OptObjValues, label="Smallest Objective Values")
plt.tick_params(labelsize=16)
plt.legend(prop=font2)
plt.show()
### demonstrate the sensitivity of dc. END

# ##Compute the point of intersection by Talor expansion 2020-4-26 AlanXu@IMI,GUES
# MRow = 2
# NRow = 3
# baseM = baseMat(D2)
# bMrow2o = baseM[MRow, ]
# bMrow2 = np.append(np.append(bMrow2o[0:MRow], bMrow2o[MRow+1:NRow]),bMrow2o[NRow+1:])###remove the equal bases
#
# bMrow3o = baseM[NRow, ]
# bMrow3 = np.append(np.append(bMrow3o[0:MRow], bMrow3o[MRow+1:NRow]),bMrow3o[NRow+1:])
#
# DiffInd = bMrow2 - bMrow3  ###compare first , then extract the meaningful indecs
# greaterInds = DiffInd > 0
# if len(bMrow2[greaterInds]) == 0 or len(bMrow2[greaterInds]) == len(bMrow2):
#     print("Can not intersect.")
# else:
#     pointResult = findIntersectP(bMrow2, bMrow3)
#
#
#
# plt.plot(dcs, rho2Vec, '--', label="rho2Vec")
# #plt.subplot(222)
# plt.plot(dcs, rho3Vec, label="rho3Vec")
# plt.legend()
# plt.show()

####Deltasum versus dc
# maxDist = int(D2.max())
# dcV = np.arange(1, maxDist)
# deltaSumV = np.zeros(len(dcV))
# for ind in range(0,len(dcV)):
#     rhotemp = localDensity(D2, dcV[ind])
#     delta_t, zeta_t = DeltaZeta(D2, rhotemp)
#     deltaSumV[ind] = sum(delta_t)
#
# plt.plot(dcV, deltaSumV)
# plt.show()

### Draw some figures
# plt.subplot(221)
# plt.plot(dcs, DiffNum)
# # plt.xlabel("dc")
# # plt.ylabel("Different Number from dc=15")
# plt.title("(a) Zeta change against dc", y = -0.2)
# plt.subplot(222)
# plt.plot(dcs, DistCost)
# # plt.xlabel("dc")
# # plt.ylabel("distance cost")
# plt.title("(b) distance cost against dc", y = -0.2)
# plt.show()
#print(zeta)

### show the trend of Gaussian Kernel
#x = np.linspace(0, 5,100)
# power= -(x ** 2)
# y= np.exp(power)
# plt.plot(x, y)
# plt.show("exp")
###Matrix divided by a colum array
# A = np.array([[2, 3], [4, 9]])
# b = np.array([[2], [3]])
# print(A/b)

##### From nneigh (\zeta) array to Adjacent List
