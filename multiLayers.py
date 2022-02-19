from numpy import e
from experiments import Experiments
from utilities import *
import os

## Random dataset example
def experiment01(experimentIndex, layer_sizes):
    # conditions
    radius = [0.001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 5
    stop_conditions = [1e-6, 1, 3000, 4] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = False

    # input = [5.1,3.5,1.4,0.2]
    input = [1 for i in range(layer_sizes[0])]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
    ept.loadSavedNetwork(folderName="MultiLayers", networkName=str(experimentIndex))
    ept.intervalBisect_ReLU_Cpp()
    # ept.trainNN()

    # ept.intervalBisect_ReLU()
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = "Experiment" + str(experimentIndex) + "\n"
    # results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)


# inputSize = 4
# experiment01(experimentIndex=1, layer_sizes=[inputSize, 10, 2])

# experiment01(experimentIndex=2, layer_sizes=[inputSize, 10, 10, 2])

# experiment01(experimentIndex=3, layer_sizes=[inputSize, 10, 10, 10, 2])

# experiment01(experimentIndex=4, layer_sizes=[inputSize, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=5, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=6, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=7, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=8, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=9, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2])

# experiment01(experimentIndex=10, layer_sizes=[inputSize, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2])

os.system("./IntervalBisect/MultiLayers")