import numpy as np

from experiments import Experiments
from utilities import *


def experiment01(experimentIndex, name):
    radius = [1e-7, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 5
    stop_conditions = [1e-6, 10, 10000, 3] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True
    layer_sizes = [10, 5, 2]
    input = [1 for i in range(layer_sizes[0])]
    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector, isSigmoid=True)

    # ept.trainNN()

    intervalResult = ept.intervalBisect_Sigmoid()

    plotSize = 51
    Nb = [i for i in range(1, plotSize)] + [i for i in range(plotSize, plotSize*3, 5)]
    Ns = [i for i in range(1, plotSize)] + [i for i in range(plotSize, plotSize*3, 5)]

    cleverResults = []
    intervalResultsLow = []
    intervalResultsUp = []
    for b in Nb:
        tmp = []
        tmp2 = []
        tmp3 = []
        for s in Ns:
            tmp.append(ept.Clever(b, s))
            tmp2.append(intervalResult[0][0])
            tmp3.append(intervalResult[0][1])
        cleverResults.append(tmp)
        intervalResultsLow.append(tmp2)
        intervalResultsUp.append(tmp3)

    path = "Saved/convergency/E" + str(experimentIndex)
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(path+"/cleverResults.csv", 'ab') as f:
        np.savetxt(f, cleverResults, delimiter=",")

    print(intervalResult)

    # plotConvergency(experimentIndex, Nb, Ns, cleverResults, intervalResultsLow, intervalResultsUp, name)
    # ept.saveNetwork(folderName="convergency", networkName=str(experimentIndex))

def experiment02(experimentIndex, name):
    radius = [1e-7, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 5
    stop_conditions = [1e-6, 20, 10000, 10] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True
    layer_sizes = [2, 2, 2]
    input = [5.1,3.5]
    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)

    ept.loadSavedNetwork("convergency", str(1))
    ept.nearBoundary()

    intervalResult = ept.intervalBisect_ReLU()
    print(intervalResult)

    plotSize = 141
    Nb = [135, 140, 145]
    Ns = [i for i in range(1, plotSize)] # + [i for i in range(plotSize, plotSize*3, 5)]

    cleverResults = []
    intervalResultsLow = []
    intervalResultsUp = []
    for b in Nb:
        tmp = []
        tmp2 = []
        tmp3 = []
        for s in Ns:
            tmp.append(ept.Clever(b, s))
            tmp2.append(intervalResult[0][0])
            tmp3.append(intervalResult[0][1])
        cleverResults.append(tmp)
        intervalResultsLow.append(tmp2)
        intervalResultsUp.append(tmp3)

    path = "Saved/convergency/E" + str(experimentIndex)
    if not os.path.exists(path):
        os.makedirs(path)

    ept.saveNetwork(folderName="convergency", networkName=str(experimentIndex))
    
    with open(path+"/cleverResults.csv", 'ab') as f:
        np.savetxt(f, cleverResults, delimiter=",")

    print(intervalResult)

    # plotConvergency(experimentIndex, Nb, Ns, cleverResults, intervalResultsLow, intervalResultsUp, name)


experimentIndex = 4
name = "noBoundary"
experiment01(experimentIndex, name)