import numpy as np
import os

from experiments import Experiments
from utilities import *

###########################################################################################################
## Random network examples
def RomdomExperiment01(experimentIndex, inputSize, hiddenSize):
    radius = [0.00000001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 2
    stop_conditions = [1e-6, 2, 10000, 10] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True

    layer_sizes = [inputSize, hiddenSize, 2]
    input = [0.5 for x in range(inputSize)]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
    # ept.loadSavedNetwork("MyRandom", str(experimentIndex))
    ept.nearBoundary_move(0)

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    ept.intervalBisect_ReLU_Cpp()
    # ept.lipMIP()
    # ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    # results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    # results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    # results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    # results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    # results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    # results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    ept.saveNetwork(folderName="Widths", networkName=str(experimentIndex))
    ept.storeExperiment(folderName="Widths")



###########################################################################################################
###########################################################################################################

for i in range(10):
    hiddenNode = pow(2, i+1)
    print("The number of hidden node is " + str(hiddenNode))
    RomdomExperiment01(experimentIndex = 1, inputSize = 2, hiddenSize = hiddenNode)
    print("##########################################################################\n")
