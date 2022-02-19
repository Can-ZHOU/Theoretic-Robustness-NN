import os
from interval import interval, imath, inf
import numpy as np
import pickle

class Interval_ReLU:
    def __init__(self, x:interval):
        self.x = x

    def ReLU(self):
        if interval[0] > self.x:
            return interval[0]
        else:
            return interval[0, inf] & self.x
        # extrema = self.x.extrema
        # if (len(extrema) > 1):
        #     left = extrema[0][0]
        #     right = extrema[1][0]
        #     if (left > 0):
        #         return self.x
        #     elif (right < 0):
        #         return interval[0]
        #     else:
        #         return interval[0, right]
        # else:
        #     value = self.x.extrema[0][0]
        #     if value > 0:
        #         return self.x
        #     else:
        #         return interval[0]

    def dReLU(self):
        extrema = self.x.extrema
        if (len(extrema) > 1):
            left = extrema[0][0]
            right = extrema[1][0]
            if (left > 0):
                return interval[1]
            elif (right < 0):
                return interval[0]
            else:
                return interval[0, 1]
        else:
            value = self.x.extrema[0][0]
            if value > 0:
                return interval[1]
            elif value < 0:
                return interval[0]
            else:
                return interval[0, 1]


class Interval_Sigmoid:
    def __init__(self, x:interval):
        self.x = x
    
    def Sigmoid(self):
        return (1 / (1 + imath.exp(-1 * self.x)))

    def dSigmoid(self):
        return self.Sigmoid() * (1 - self.Sigmoid())



#########################################
## Some useful interval helper functions
#########################################

# floating -> interval without redius
def convert_array(a):
    result = []
    for i in range(len(a)):
        result.append(interval[a[i]])
    return result

def convert_2D_matrix(m):
    result = []
    for i in range(len(m)):
        tmp = []
        for j in range(len(m[i])):
            tmp.append(interval[m[i][j]])
        result.append(tmp)
    return result

# floating -> interval with redius
def convert_array_radius(a, radius):
    result = []
    for i in range(len(a)):
        result.append(interval[a[i]-radius, a[i]+radius])
    return result

def convert_2D_matrix_radius(m, radius):
    result = []
    for i in range(len(m)):
        tmp = []
        for j in range(len(m[i])):
            tmp.append(interval[m[i][j]-radius, m[i][j]+radius])
        result.append(tmp)
    return result

# repeat list: [[a], [b]] -> [[a], [a], [b], [b]]
def repearList(I_X):
    result = []
    for x in I_X:
        result.append(x.copy())
        result.append(x.copy())
    return result

# 1D array -> 2D array
def getBisectionList(I_X):
        new_X = []

        for x in I_X:
            extrema = x.extrema
            if (len(extrema) > 1):
                left = extrema[0][0]
                right = extrema[1][0]
                medium = (left + right) / 2
                part01 = interval[left, medium]
                part02 = interval[medium, right]

                if not new_X:
                    new_X.append([part01])
                    new_X.append([part02])
                else:
                    new_X = repearList(new_X)
                    for i in range(0, len(new_X), 2):
                        new_X[i].append(part01)
                        new_X[i+1].append(part02)
            else:
                print("Something is wrong in getBisectionList.")
                exit(-1)

        return new_X


#########################################
## Some useful floating helper functions
#########################################

def storeParameters(model, x, index):
    need = []
    for name, param in model.named_parameters():
        # print(name, param.size())
        need.append(param.data)

    W1 = need[0].data.numpy()
    b1 = need[1].data.numpy()
    W2 = need[2].data.numpy()

    # Saving as csv
    path = "parameters/E" + str(index)
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt(path+"/X0.csv", [x], delimiter=",")
    np.savetxt(path+"/Pred_X0.csv", np.array([0]), delimiter=",")
    np.savetxt(path+"/W1.csv", W1, delimiter=",")
    np.savetxt(path+"/W2.csv", W2, delimiter=",")
    np.savetxt(path+"/b1.csv", [b1], delimiter=",")


def storeParametersMulti(model, x, index):
    # Saving as csv
    path = "parameters/MultiLayers/E" + str(index)
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt(path+"/X0.csv", [x], delimiter=",")

    with open(path+"/weights.csv", 'ab') as fw:
        with open(path+"/bias.csv", 'ab') as fb:
            for name, param in model.named_parameters():
                tmp = param.data.numpy()
                if "weight" in name:
                    np.savetxt(fw, tmp, delimiter=",")
                    np.savetxt(fw, ["###"], delimiter="", fmt="%s")

                if "bias" in name:
                    np.savetxt(fb, [tmp], delimiter=",")
