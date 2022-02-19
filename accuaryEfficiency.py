import numpy as np
import os

from experiments import Experiments
from utilities import *

###########################################################################################################
## Random network examples
def RomdomExperiment01(experimentIndex):
    radius = [0.0000001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 2
    stop_conditions = [1e-6, 20, 3000, 10] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True

    layer_sizes = [2, 2, 2]
    input = [0.5, 0.5]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
    ept.loadSavedNetwork("AccuaryEfficiency", str(1))
    ept.nearBoundary()

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    ept.intervalBisect_ReLU_Cpp()
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    # ept.saveNetwork(folderName="AccuaryEfficiency", networkName=str(experimentIndex))
    # ept.storeExperiment(folderName="AccuaryEfficiency")

def RomdomExperiment02(experimentIndex):
    radius = [0.0000001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 2
    stop_conditions = [1e-6, 2, 3000, 2] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True

    layer_sizes = [2, 2, 2]
    input = [0.5, 0.5]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
    ept.loadSavedNetwork("AccuaryEfficiency", str(2))
    ept.nearBoundary()

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    ept.intervalBisect_ReLU_Cpp()
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    # ept.saveNetwork(folderName="AccuaryEfficiency", networkName=str(experimentIndex))
    # ept.storeExperiment(folderName="AccuaryEfficiency")

def RomdomExperiment03(experimentIndex):
    radius = [0.0000001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 2
    stop_conditions = [1e-6, 20, 3000, 10] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = True

    layer_sizes = [2, 2, 2]
    input = [0.5, 0.5]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)
    ept.loadSavedNetwork("AccuaryEfficiency", str(3))
    ept.nearBoundary()

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    ept.intervalBisect_ReLU_Cpp()
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    # ept.saveNetwork(folderName="AccuaryEfficiency", networkName=str(experimentIndex))
    # ept.storeExperiment(folderName="AccuaryEfficiency")

###########################################################################################################
## IRIS example
def IRISExperiment(experimentIndex):
    # conditions
    radius = [0.001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0, 0]
    dataIndex = 0
    stop_conditions = [1e-6, 1, 3000, 4] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = False

    layer_sizes = [4, 3, 3]
    input = [5.1,3.5,1.4,0.2]

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)

    # ept.trainNN()
    ept.loadSavedNetwork("AccuaryEfficiency", str(4))

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    ept.intervalBisect_ReLU_Cpp()
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    # ept.saveNetwork(folderName="AccuaryEfficiency", networkName=str(experimentIndex))
    # ept.storeExperiment(folderName="AccuaryEfficiency")

###########################################################################################################
## MNIST example
def MNISTExperiment(experimentIndex):
    # conditions
    radius = [0.001, 0, 0] # [input, weight, bias]
    outputIndex = 0
    c_vector = [1, 0]
    dataIndex = 2
    stop_conditions = [1e-6, 1, 3000, 4] #[minGap, maxiteration, maxBoxes, maxSame]
    needBisect = False

    layer_sizes = [784, 10, 10, 2]
    X_train, X_test, Y_train, Y_test = loadData(dataIndex, 0)
    input = X_test[0].numpy().tolist()      

    ept = Experiments(radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector)

    # ept.trainNN()
    ept.loadSavedNetwork("AccuaryEfficiency", str(5))

    ept.intervalBisect_ReLU()
    print("interval Cpp:")
    print(os.system("./IntervalBisect/MNIST01"))
    ept.lipMIP()
    ept.ZLip()
    ept.OtherMethods()

    results = ""
    results += "interval ReLU: value - " + str(ept.interval_result_ReLU) + " timing - " + str(ept.interval_time_ReLU) + "\n"
    results += "lipMIP ReLU: value - " + str(ept.lipMIP_result) + " timing - " + str(ept.lipMIP_time) + "\n"
    results += "ZLip ReLU: value - " + str(ept.ZLip_result) + " timing - " + str(ept.ZLip_time) + "\n"
    results += "CLEVER ReLU: value - " + str(ept.otherMethodsResults[0][1]) + " timing - " +  str(ept.otherMethodsResults[0][0]) + "\n"
    results += "FastLip ReLU: value - " + str(ept.otherMethodsResults[1][1]) + " timing - " + str(ept.otherMethodsResults[1][0]) + "\n"
    results += "NaiveUB ReLU: value - " + str(ept.otherMethodsResults[2][1]) + " timing - " + str(ept.otherMethodsResults[2][0]) + "\n"
    results += "RandomLB ReLU: value - " + str(ept.otherMethodsResults[3][1]) + " timing - " + str(ept.otherMethodsResults[3][0]) + "\n"
    results += "SeqLip ReLU: value - " + str(ept.otherMethodsResults[4][1]) + " timing - " + str(ept.otherMethodsResults[4][0]) + "\n\n"

    print(results)
    # ept.saveNetwork(folderName="AccuaryEfficiency", networkName=str(experimentIndex))
    # ept.storeExperiment(folderName="AccuaryEfficiency")



###########################################################################################################
###########################################################################################################
RomdomExperiment01(experimentIndex = 10)
RomdomExperiment02(experimentIndex = 10)
RomdomExperiment03(experimentIndex = 10)
IRISExperiment(experimentIndex = 10)
MNISTExperiment(experimentIndex = 10)