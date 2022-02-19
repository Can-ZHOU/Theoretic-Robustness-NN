import numpy as np

from experiments import Experiments
from utilities import *

radius = [0.001, 0, 0] # [input, weight, bias]
outputIndex = 0
stop_conditions = [1e-6, 1, 3000, 4] #[minGap, maxiteration, maxBoxes, maxSame]
needBisect = False
experimentIndex = 10
dataIndex=0
c_vector=[1, 0]


layer_sizes = [2, 2, 2]
radius_list = np.arange(0, 0.01, 1e-4).tolist()
offset_list = np.arange(0, 0.02, 1e-2).tolist()
# radius_list = np.arange(0, 0.1, 0.02).tolist()
# offset_list = np.arange(0, 0.1, 0.03).tolist()
input = [1, 1]

ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
ept.loadSavedNetwork("WeightBias", str(8))
inputNearBoundary = ept.nearBoundary()

results = []
for offset in offset_list:
    operations = [1, 0, -1]
    tmp = []
    for i in range(3):
        for j in range(3):
            inputNew = [(inputNearBoundary[0]+operations[i]*offset), (inputNearBoundary[1]+operations[j]*offset)]
            ept.setInput(inputNew)
            tmpResult = ept.weightBiasExperiment(radius_list)
            tmp.append(tmpResult)
    results.append(tmp)

ept.saveNetwork(folderName="WeightBias", networkName=str(experimentIndex))

plotAll(results, radius_list, experimentIndex)