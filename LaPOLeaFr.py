import numpy as np
import queue as que
import random
class LaPOLeaF():

    def ConstructOLeaF(self, Zeta, Centers):
        ND = len(Zeta)
        Zeta[Centers] = -1
        AL = [np.zeros((1, 1))-1 for i in range(ND) ] ## Begining with -1 indicates that the List is Empty!!
        for i in range(0, ND):
            if Zeta[i] > -1:
                ind = Zeta[i]
                temp = np.append(AL[ind], i)
                AL[ind] = temp

        for i in range(0, ND):
            S = np.array(AL[i]).shape
            if S != (1, 1):
                AL[i] = np.append(AL[i][1:], -1)  ### move -1 to the end

        return AL

    ####Find the layer Index for each subtree in OLeaF: 2020-5-1
    def LayerInd(self, AL, Root):
        layerInds = {Root: 1}
        Q = que.Queue()
        Q.put(Root)
        while not Q.empty():
            QHead = int(Q.get())
            if AL[QHead][0] > -1:
                for child in AL[QHead]:
                    if child > -1:
                        Q.put(child)
                        layerInds[child] = layerInds[QHead]+1
        return layerInds

    ###Initialize the labeled data
    def InitLabels( self, AllLabels, K, LAmount):
        ND = len(AllLabels)
        KnownLabelsInds = np.zeros(LAmount*K)-1
        for k in range(K):###Labels starts at 0
            curClassInds =  [i for i in range(ND) if AllLabels[i] == k]
            random.shuffle(curClassInds)###shuffle the target array, no returned values
            tempLabeledInds = curClassInds[0:LAmount]
            KnownLabelsInds[k*LAmount:(k+1)*LAmount] = tempLabeledInds

        AllLableVec = np.zeros((ND, K))
        for i in range(LAmount*K):
            currInd = int(KnownLabelsInds[i])
            cl = int(AllLabels[currInd])
            AllLableVec[currInd, cl] = 1

        return AllLableVec


    ###Propagation C2P, from the angle of root.
    def c2pPropagation(self, AL, layerInds, CurrentLabels, delta):
        maxLayerInd= max(layerInds)
        ND = len(layerInds)
        for LInd in range(maxLayerInd-1, 0, -1):## for each layer of all subtrees propagate simultaneously
            currentNodes = [i for i in range(ND) if layerInds[i] == LInd] ## for each root of the atom tree
            for nodeInd in range(len(currentNodes)): ## for each child w.r.t. the current root
                node = currentNodes[nodeInd]
                children = AL[node]
                if max(CurrentLabels[node, :]) == 1 or children[0] == -1:###labeled or has no children
                    continue
                PropaWeightSum = 0
                for childI in range(len(children)):
                    Child = int(children[childI])
                    if sum(CurrentLabels[Child, :]) > 0:
                        Weight = 1 / delta[Child]
                        CurrentLabels[node, :] += Weight * CurrentLabels[Child, :]
                        PropaWeightSum = PropaWeightSum + Weight
                if PropaWeightSum > 0:
                    CurrentLabels[node, :] /= PropaWeightSum
        return CurrentLabels

    def r2rPropagation(self, labelC2P, DistCenters, rho, Centers):##Alanxu @ 2020-5-17 untested!
        r2rLabels = labelC2P
        WholeRoot = Centers[0]
        CenterNum = len(Centers)

        CenterDistMin = 100000
        if sum(labelC2P[WholeRoot]) == 0: ### Label the whole root if it is not labeled,
             for i in range(1, CenterNum):
                 if  sum(labelC2P[Centers[i]]) > 0 and DistCenters[0, i] < CenterDistMin:
                    r2rLabels[WholeRoot, :] = labelC2P[Centers[i], :]
                    CenterDistMin = DistCenters[0, i]

        unlabeledCenters = [Centers[i] for i in range(0, CenterNum) if sum(r2rLabels[Centers[i], :]) == 0]
        labeledCenters = list(set(Centers)-set(unlabeledCenters))
        for uCentI in unlabeledCenters:
            CenterDistMin2 = 100000
            for lCentI in labeledCenters:
                uICenters = (Centers.tolist()).index(uCentI)
                lICenters = (Centers.tolist()).index(lCentI)
                if rho[lCentI] > rho[uCentI] and DistCenters[uICenters, lICenters]< CenterDistMin2:
                    r2rLabels[uCentI, :] = r2rLabels[lCentI, :]
                    CenterDistMin2 = DistCenters[uICenters, lICenters]
        return r2rLabels

    def p2cPropagation(self, AL, layerInds, CurrentLabels, delta):
        ND = len(layerInds)
        K = (np.shape(CurrentLabels))[1]
        finalLabels = CurrentLabels
        LabeledFlag = [sum(CurrentLabels[i,:])>0 for i in range(ND) ]

        MaxDepth = max(layerInds)
        for Depth in range(1, MaxDepth):
            currentParents = [i for i in range(ND) if layerInds[i] == Depth]
            for ParentInd in range(1, len(currentParents)):
                parent = currentParents[ParentInd]
                childrenF = AL[parent]
                children = [int(el) for el in childrenF]
                if children[0]== -1: # has no children
                    continue
                else:
                    labeledChildreContri = np.zeros((1, K)) ## contribution from labeled children

                    weightSum = 0
                    for jj in range(len(children)-1):
                        child = children[jj]
                        Weight = 1 / delta[child]
                        weightSum = weightSum + Weight # for both labeled and unlabeled
                        if LabeledFlag[child] == True:
                            labeledChildreContri +=  Weight * CurrentLabels[child, :]

                    for jj in range(len(children)-1):
                        child = children[jj]
                        if LabeledFlag[child] == False:# is an unlabeled child
                            finalLabels[child, :] = finalLabels[parent, :] - labeledChildreContri / weightSum
        return finalLabels