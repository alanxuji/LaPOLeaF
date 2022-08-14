import numpy as np
import collections


def GetSublt(Pa, AL, Pnode):
    queue = collections.deque()
    queue.append(Pnode)

    while len(queue) != 0:
        Node = queue.popleft()
        indes = [i for i, x in enumerate(Pa) if x == Node]

        for ind in indes:
            AL = np.append(AL, ind)
            queue.append(ind)
    return AL


def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    # print('*' * 30, '\n compute distance matrix...')
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return np.sqrt(sqdist)


class LeadingTree:
    """
    Leading Tree
    """
    def __init__(self, X_train, dc, lt_num):
        self.X_train = X_train
        self.dc = dc
        self.lt_num = lt_num
        self.D = EuclidianDist2(X_train, X_train)  # Calculate the distance matrix D
        self.density = None
        self.Pa = None
        self.delta = None
        self.gamma = None
        self.gamma_D = None
        self.Q = None
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i] store all indexes of a subtree
        self.layer = np.zeros(len(X_train), dtype=int)

    def ComputeLocalDensity(self, D, dc):
        """
        Calculate the local density of samples
        :param D: The Euclidean distance of all samples
        :param dc:Bandwidth parameters
        :return:
        self.density: local density of all samples
        self.Q: Sort the density index in descending order
        """
        # print('*'*30,'\n compute density...')
        tempMat1 = np.exp(-(D ** 2))
        tempMat = np.power(tempMat1, dc ** (-2))
        self.density = np.sum(tempMat, 1) - 1
        self.Q = np.argsort(self.density)[::-1]

    def ComputeParentNode(self, D, Q):
        """
        Calculate the distance to the nearest data point of higher density (delta) and the parent node (Pa)
        :param D: The Euclidean distance of all samples
        :param Q:Sort by index in descending order of sample local density
        :return:
        self.delta: the distance of the sample to the closest data point with a higher density
        self.Pa: the index of the parent node of the sample
        """
        # print('*' * 30, '\n compute leading nodes...')
        self.delta = np.zeros(len(Q))
        self.Pa = np.zeros(len(Q), dtype=int)
        for i in range(len(Q)):
            if i == 0:
                self.delta[Q[i]] = max(D[Q[i]])
                self.Pa[Q[i]] = -1
            else:
                greaterInds = Q[0:i]
                D_A = D[Q[i], greaterInds]
                self.delta[Q[i]] = min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]

    def ProCenter(self, density, delta, Pa):
        """
        Calculate the probability of being chosen as the center node and Disconnect the Leading Tree
        :param density: local density of all samples
        :param delta: the distance of the sample to the closest data point with a higher density
        :param Pa: the index of the parent node of the sample
        :return:
        self.gamma: the probability of the sample being chosen as a center node
        self.gamma_D: Sort the gamma index in descending order
        """
        self.gamma = density * delta
        self.gamma_D = np.argsort(self.gamma)[::-1]

        # Disconnect the Leading Tree
        for i in range(self.lt_num):
            Pa[self.gamma_D[i]] = -1

    def GetSubtree(self, gamma_D, lt_num):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        # print('*' * 30, '\n retrieve the subtrees...')
        for i in range(lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])  # center node of subtrees
            self.AL[i] = GetSublt(self.Pa, self.AL[i], gamma_D[i])  # get the whole subtree by the center node

    def GetLayer(self, Pnode, layer, l):
        """
        Calculate nodes' layer
        :param Pnode: center nodes
        :param layer: samples' layer
        :param l: current layer
        :return:
        self.layer: layer of all samples
        """
        # print('*' * 30, '\n compute layer index...')
        queue = collections.deque()
        queue.append(Pnode)
        while len(queue) != 0:
            Node = queue.popleft()
            indes = [i for i, x in enumerate(self.Pa) if x in Node]
            layer[indes] = l
            if len(indes) != 0:
                queue.append(indes)
            l = l + 1

    def Edges(self, Pa):  # store edges of subtrees
        """

        :param Pa:  the index of the parent node of the sample
        :return:
        self. edges: pairs of child node and parent node
        """
        edgesO = np.array(list(zip(range(len(Pa)), Pa)))
        ind = edgesO[:, 1] > -1
        self.edges = edgesO[ind,]

    def fit(self):
        self.ComputeLocalDensity(self.D, self.dc)
        self.ComputeParentNode(self.D, self.Q)
        self.ProCenter(self.density, self.delta, self.Pa)
        self.GetSubtree(self.gamma_D, self.lt_num)
        self.Edges(self.Pa)

        Pnodes = np.zeros(0, dtype=int)  # center nodes
        for i in range(len(self.AL)):
            Pnodes = np.append(Pnodes, self.AL[i][0])
        self.GetLayer(Pnodes, self.layer, 1)
        self.layer = self.layer + 1
