import os
import time
from utilities import loadData
import torch
import torch.nn as nn
import numpy as  np
from art.estimators.classification import PyTorchClassifier

from IntervalBisect.intervalBisect import IntervalBisect
from OtherMethods.lipMIP.relu_nets import ReLUNet
from OtherMethods.lipMIP.lipMIP import LipProblem
from OtherMethods.lipMIP.hyperbox import Hyperbox as lipMIP_Hyperbox
from OtherMethods.ZLip.z_lip import ZLip
from OtherMethods.ZLip.hyperbox import Hyperbox as ZLip_Hyperbox
from OtherMethods.ZLip.zonotope import Zonotope
from OtherMethods.other_methods import CLEVER, FastLip, NaiveUB, RandomLB, SeqLip
from OtherMethods.CLEVER import clever_modified



class Experiments:
    def __init__(self, radius, layer_sizes, outputIndex, stop_conditions, needBisect, experimentIndex, input, dataIndex, c_vector, isSigmoid=False):
        self.experimentIndex = experimentIndex
        self.radius = radius
        self.radius_inputs = radius[0]
        self.outputIndex = outputIndex
        self.inputs_size = layer_sizes[0]
        self.output_size = layer_sizes[len(layer_sizes)-1]
        self.layer_sizes = layer_sizes
        self.network = ReLUNet(layer_sizes=layer_sizes, bias=True, isSigmoid=isSigmoid)
        self.interval_result_ReLU = 0.0
        self.interval_result_Sigmoid = 0.0
        self.lipMIP_result = 0.0
        self.ZLip_result = 0.0
        self.interval_time_ReLU = 0
        self.interval_time_Sigmoid = 0
        self.lipMIP_time = 0
        self.ZLip_time = 0
        self.input = input
        self.stop_conditions = stop_conditions
        self.needBisect = needBisect
        self.c_vector = np.array(c_vector)
        self.dataIndex = dataIndex
        self.randomSeed = 0
        self.otherMethodsResults = []
        self.allResults = ""

    def setInput(self, input):
        self.input = input

    def trainNN(self):
        # read the data
        X_train, X_test, Y_train, Y_test = loadData(self.dataIndex, self.randomSeed)

        # train
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        for epoch in range(1000):
            optimizer.zero_grad()
            out = self.network(X_train)
            loss = criterion(out, Y_train)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("epoch", epoch, "loss", loss.data)

        # test
        predict_out =self.network(X_test)
        _, Y_predict = torch.max(predict_out, 1)
        Y_predict.tolist()
        Y_test.tolist()

        count = 0
        for i in range(len(Y_predict)):
            if Y_predict[i] == Y_test[i]:
                count += 1
        accuracy = count/len(Y_test)
        print("The accuracy of trained networkwork is ", accuracy)

    def nearBoundary(self):
        need = []
        for name, param in self.network.named_parameters():
            need.append(param.data)

        W1 = need[0].data.numpy()
        b1 = need[1].data.numpy()
        dataLength = self.layer_sizes[0]
        A = np.zeros((dataLength, dataLength))
        B = np.zeros(dataLength)
        bias = -1 * b1

        for i in range(dataLength):
            B[i] = bias[i]
            for j in range(dataLength):
                A[i][j] = W1[i][j]

        self.input = np.linalg.solve(A, B).tolist()
        return self.input

    def saveNetwork(self, folderName, networkName):
        path = "Saved/"+ folderName + "/E" + str(self.experimentIndex)
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save(self.network, path + "/" + networkName + ".pt")

    def loadSavedNetwork(self, folderName, networkName):
        path = "Saved/"+ folderName + "/E" + str(networkName)
        self.network = torch.load(path + "/" + networkName + ".pt")

    def storeExperiment(self, folderName):
        path = "Saved/"+ folderName + "/E" + str(self.experimentIndex)
        if not os.path.exists(path):
            os.makedirs(path)

        # save network parameters
        with open(path+"/weights.csv", 'ab') as fw:
            with open(path+"/bias.csv", 'ab') as fb:
                for name, param in self.network.named_parameters():
                    tmp = param.data.numpy()
                    if "weight" in name:
                        np.savetxt(fw, tmp, delimiter=",")
                        np.savetxt(fw, [[]], delimiter=",")

                    if "bias" in name:
                        np.savetxt(fb, tmp, delimiter=",")
                        np.savetxt(fb, [[]], delimiter=",")

        # save other experiment info
        otherInfo = ""
        otherInfo += 'experimentIndex: ' + str(self.experimentIndex) + "\n"
        otherInfo += 'radius_inputs: ' + str(self.radius_inputs) + "\n"
        otherInfo += 'interval_result_ReLU: ' + str(self.interval_result_ReLU) + "\n"
        otherInfo += 'interval_result_Sigmoid: ' + str(self.interval_result_Sigmoid) + "\n"
        otherInfo += 'lipMIP_result: ' + str(self.lipMIP_result) + "\n"
        otherInfo += 'ZLip_result: ' + str(self.ZLip_result) + "\n"
        otherInfo += 'interval_time_ReLU: ' + str(self.interval_time_ReLU) + "\n"
        otherInfo += 'interval_time_Sigmoid: ' + str(self.interval_time_Sigmoid) + "\n"
        otherInfo += 'lipMIP_time: ' + str(self.lipMIP_time) + "\n"
        otherInfo += 'ZLip_time: ' + str(self.ZLip_time) + "\n"
        otherInfo += 'otherMethodsResults: '+ str(self.otherMethodsResults) + "\n"
        otherInfo += 'dataIndex: ' + str(self.dataIndex) + "\n"
        otherInfo += 'randomSeed: ' + str(self.randomSeed) + "\n"
        otherInfo += 'input: ' + str(self.input) + "\n"
        otherInfo += 'layer_sizes: '+ str(self.layer_sizes) + "\n"
        otherInfo += 'stop_conditions: '+ str(self.stop_conditions) + "\n"
        otherInfo += 'needBisectect: '+ str(self.needBisect) + "\n"

        allResults = ""
        allResults += "interval ReLU: value - " + str(self.interval_result_ReLU)
        allResults += " timing - " + str(self.interval_time_ReLU) + "\n" 
        allResults += "lipMIP ReLU: value - " + str(self.lipMIP_result) + " timing - " + str(self.lipMIP_time) + "\n"  
        allResults += "ZLip ReLU: value - " + str(self.ZLip_result) + " timing - " + str(self.ZLip_time) + "\n"  
        allResults += "CLEVER ReLU: value - " + str(self.otherMethodsResults[0][1]) + " timing - " +  str(self.otherMethodsResults[0][0]) + "\n"  
        allResults += "FastLip ReLU: value - " + str(self.otherMethodsResults[1][1]) + " timing - " + str(self.otherMethodsResults[1][0]) + "\n"  
        allResults += "NaiveUB ReLU: value - " + str(self.otherMethodsResults[2][1]) + " timing - " + str(self.otherMethodsResults[2][0]) + "\n"  
        allResults += "RandomLB ReLU: value - " + str(self.otherMethodsResults[3][1]) + " timing - " + str(self.otherMethodsResults[3][0]) + "\n"  
        allResults += "SeqLip ReLU: value - " + str(self.otherMethodsResults[4][1]) + " timing - " + str(self.otherMethodsResults[4][0])

        otherInfo += allResults
        self.allResults = allResults
        with open(path+"/otherInfo.txt", "w") as text_file:
            text_file.write(otherInfo)
        
    
    def intervalBisect_ReLU(self):
        start = time.time_ns()
        intervalResults = IntervalBisect(self.layer_sizes, self.input, self.outputIndex, self.radius, self.network, self.stop_conditions, self.needBisect)
        self.interval_result_ReLU = intervalResults.compute_ReLU()
        end = time.time_ns()
        self.interval_time_ReLU = end - start
        return self.interval_result_ReLU

    def intervalBisect_ReLU_Cpp(self):
        intervalResults = IntervalBisect(self.layer_sizes, self.input, self.outputIndex, self.radius, self.network, self.stop_conditions, self.needBisect)
        intervalResults.compute_ReLU_cpp(self.experimentIndex, isMultiLayers=False)

    def intervalBisect_Sigmoid(self):
        self.interval_result_Sigmoid = IntervalBisect(self.layer_sizes, self.input, self.outputIndex, self.radius, self.network, self.stop_conditions, self.needBisect).compute_Sigmoid()
        return self.interval_result_Sigmoid
    
    def lipMIP(self):
        start = time.time_ns()
        hbox = lipMIP_Hyperbox.build_customized_hypercube(self.input, self.radius_inputs)
        domain = hbox
        myLipMIP = LipProblem(self.network, domain, self.c_vector)
        tmp =  myLipMIP.compute_max_lipschitz().shrink()
        self.lipMIP_result = tmp.value
        end = time.time_ns()
        self.lipMIP_time = end - start
        return self.lipMIP_result
    
    def ZLip(self):
        start = time.time_ns()
        primal_norm = 'linf'
        hbox = ZLip_Hyperbox.build_customized_hypercube(self.input, self.radius_inputs)
        domain = Zonotope.from_hyperbox(hbox)
        myLipMIP = ZLip(self.network, self.c_vector, domain, primal_norm)
        self.ZLip_result = myLipMIP.compute().item()
        end = time.time_ns()
        self.ZLip_time = end - start
        return self.ZLip_result

    def Clever(self, Nb, Ns):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.net.parameters(), lr=0.01)
        classifier = PyTorchClassifier(
                model=self.network.net,
                loss=criterion,
                optimizer=optimizer,
                input_shape=(1, self.inputs_size),
                nb_classes=self.output_size,
            )
        pool_size = 2
        result = clever_modified(classifier, np.array(self.input), Nb, Ns, self.radius_inputs, norm=np.inf, pool_factor=pool_size, outputIndex = self.outputIndex)
        return result

    def OtherMethods(self):
        primal_norm = 'linf'
        c_vector = torch.Tensor(self.c_vector)

        otherMethodsResults = []

        for other_method in [CLEVER, FastLip, NaiveUB, RandomLB, SeqLip]:
            start = time.time_ns()
            domain = lipMIP_Hyperbox.build_customized_hypercube(self.input, self.radius_inputs)
            test_object = other_method(self.network, c_vector, domain=domain, primal_norm=primal_norm)
            test_object.compute()
            # print(other_method.__name__ + ' ran in %f seconds and has value %f' % 
            #       (test_object.compute_time, test_object.value))
            end = time.time_ns()
            otherMethodsResults.append((end - start, test_object.value))
        
        self.otherMethodsResults = otherMethodsResults.copy()
        return self.otherMethodsResults

    def weightBiasExperiment(self, radius_list):
        results = []
        for r in radius_list:
            self.radius = [self.radius_inputs, r, r]
            result = self.intervalBisect_ReLU()
            results.append(result)
        return results