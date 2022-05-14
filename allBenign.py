from OtherMethods.lipMIP.relu_nets import ReLUNet
from IntervalBisect.intervalBisect import IntervalBisect
from OtherMethods.CLEVER import clever_modified
from art.estimators.classification import PyTorchClassifier

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as  np
import os

def plotFunction(x, y):
    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.plot(x, y, color='black', linestyle='dashed', linewidth = 2,
             marker='o', markerfacecolor='red', markersize=11)

def plotFunctionOut(x, y, fileName, nnSize):
    plt.ylim(-5,5)
    plt.xlim(-5,5)
    plt.title(str(nnSize) + ' ReLU Neurons')
    plt.plot(x, y)
    plt.savefig(fileName)
    plt.close()

def plotFunctionLip(x, y, fileName):
    plt.plot(x, y)
    plt.xlabel('Capacity')
    plt.ylabel('Lipschitz Constant')
    plt.savefig(fileName)
    plt.close()

def plotTwoLines(x, y1, y2, fileName):
    plt.plot(x, y1, label="lipDT")
    plt.plot(x, y2, label="lipDT")
    plt.legend()
    plt.savefig(fileName)
    plt.close()


def intervalBisect_ReLU(layer_sizes, input, network, radius):
    stop_conditions = [1e-6, 20, 10000, 10] #[minGap, maxiteration, maxBoxes, maxSame]
    outputIndex=0
    needBisect = True
    intervalResults = IntervalBisect(layer_sizes, input, outputIndex, radius, network, stop_conditions, needBisect)
    return intervalResults.compute_ReLU()

def Clever(network, Nb, Ns, input, radius):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0001)
    classifier = PyTorchClassifier(
        model=network.net,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 1),
        nb_classes=1,
    )
    pool_size = 2
    result = clever_modified(classifier, np.array(input), Nb, Ns, radius[0], norm=np.inf, pool_factor=pool_size, outputIndex = 0)
    return result

def loadSavedNetwork(networkName):
    path = "benign/savedNN"
    return torch.load(path + "/" + networkName + ".pt")

def saveNetwork(network, networkName, epoch):
    path = "allBenign/savedNN/" + networkName
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(network, path + "/" + networkName + "_" + str(epoch) + ".pt")

def net(x, y, nnSize, fileName):
    layer_sizes = [1, nnSize, 1]
    # network = loadSavedNetwork(fileName)
    network = ReLUNet(layer_sizes=layer_sizes, bias=True, isSigmoid=False)

    # train
    x_train = torch.from_numpy(x).float()
    y_train = torch.from_numpy(y).float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0008)
    tmp1 = tmp2 = 0
    lipDTResults = []
    cleverResults = []
    radius = [5, 0, 0] #[input, weight, bias]
    Nb = 20
    Ns = 100
    for epoch in range(2500):
        out = network(x_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        saveNetwork(network, fileName, epoch)
        # lipDT = intervalBisect_ReLU(layer_sizes, [0], network, radius)[0][1]
        # clever = Clever(network, Nb, Ns, [0], radius)
        # lipDTResults.append(lipDT)
        # cleverResults.append(clever)
        # print("lipDT score is " + str(lipDT))
        # print("clever score is " + str(clever))

        if epoch % 20 == 0:
            tmp1 = tmp2
            tmp2 = loss.data
            if((tmp1 == loss.data) and (tmp2 == loss.data)):
                break
            print("epoch", epoch, "loss", loss.data)

    # test
    # x_out = np.reshape(np.linspace(-5., 5., 1000), (-1, 1))
    # x_test = torch.from_numpy(x_out).float()
    # y_test = network(x_test)
    # y_out = torch.reshape(y_test, (-1,)).tolist()
    # plotFunction(x_orginal, y_orginal)
    # plotFunctionOut(x_out, y_out, 'benign/imgs/' + fileName + '.png', nnSize)
    return lipDTResults, cleverResults



###############################################################################
# the original function
x_orginal = range(-5,6)
y_orginal = [-2.5, -2.6, -3, 1, -0.2, 0.1, 2.5, 2.3, 2.4, -0.1, 2.7]
x = np.reshape(np.array(x_orginal), (-1, 1))
y = np.reshape(np.array(y_orginal), (-1, 1))

# nnList = [i for i in range(3, 102, 3)]
# nnList.append(1500)
# nnList.append(2000)
# nnList.append(3000)
nnList = [3000]
results = []

for item in nnList:
    fileName = 'b' + str(item)
    lipDTResults, cleverResults = net(x, y, item, fileName)
    # xArray = [i for i in range(len(lipDTResults))]
    # plotTwoLines(xArray, lipDTResults, cleverResults, 'allBenign/'+fileName+'.png')
    # network = loadSavedNetwork(fileName)
    # layer_sizes = [1, item, 1]
    # tmp = intervalBisect_ReLU(layer_sizes, [0], network)
    # print(tmp)
    # results.append(tmp[0][1])

# plotFunctionLip(nnList, results, 'benign/Lip.png')

