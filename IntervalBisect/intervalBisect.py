from interval import interval
import numpy as np
import time

from IntervalBisect.utilities import *
from IntervalBisect.IntervalCPP_ReLU_V2 import *

class IntervalBisect:
    def __init__(self, layer_sizes, input, outputIndex, radius, network, stop_conditions, needBisect):
        self.inputs_size = layer_sizes[0]
        self.output_size = layer_sizes[len(layer_sizes)-1]
        self.layer_sizes = layer_sizes
        self.input = input
        self.outputIndex = outputIndex
        self.network = network
        self.radius_inputs = radius[0]
        self.radius_weights = radius[1]
        self.radius_bias = radius[2]
        self.minGap = stop_conditions[0]
        self.maxIteration = stop_conditions[1]
        self.maxBoxes = stop_conditions[2]
        self.maxSame = stop_conditions[3]
        self.needBisect = needBisect
        self.countSame = 0
        self.previousLowMax = 0
        self.hiddenLayerNumber = len(layer_sizes) - 2
        self.I_weights = []
        self.I_bias = []
        self.I_X = convert_array_radius(self.input, self.radius_inputs)
        self.forwardResult = []
        self.boxes = []
        self.gradent = []
        self.getParameters()
        self.computTime = 0
        


    #######################################################
    # For ReLU
    #######################################################
    def get_gradient(self, I_X):
        
        result = []
        gradient = []
        for k in range(self.hiddenLayerNumber):
            tmpResult = []
            for i in range(self.layer_sizes[k+1]):
                tmp = interval(0)
                for j in range(self.layer_sizes[k]):
                    if k == 0:
                        tmp += self.I_weights[k][i][j] * I_X[j]
                    else:
                        tmp += self.I_weights[k][i][j] * Interval_ReLU(result[k-1][j]).ReLU()
                tmp += self.I_bias[k][i]
                tmpResult.append(tmp)
            result.append(tmpResult)
        self.forwardResult = result.copy()

        for k in range(1, self.hiddenLayerNumber+1):
            gradientOutput = []
            for outputIndex in range(self.layer_sizes[k+1]):
                gradientInput = []
                for inputIndex in range(self.inputs_size):
                    gradientTmp = interval(0)
                    if k == 1:
                        for i in range(self.layer_sizes[k]):
                            gradientTmp += self.I_weights[k][outputIndex][i] * Interval_ReLU(self.forwardResult[k-1][i]).dReLU() * self.I_weights[k-1][i][inputIndex]
                            
                    else:
                        for i in range(self.layer_sizes[k]):
                            gradientTmp += self.I_weights[k][outputIndex][i] * Interval_ReLU(self.forwardResult[k-1][i]).dReLU() * gradient[k-2][i][inputIndex]
                    
                    gradientInput.append(gradientTmp)
                gradientOutput.append(gradientInput)
            gradient.append(gradientOutput)
        self.gradent = gradient.copy()

    def forward_ReLU(self, I_X):
        result = []
        for k in range(self.hiddenLayerNumber):
            tmpResult = []
            for i in range(self.layer_sizes[k+1]):
                tmp = interval(0)
                for j in range(self.layer_sizes[k]):
                    if k == 0:
                        tmp += self.I_weights[k][i][j] * I_X[j]
                    else:
                        tmp += self.I_weights[k][i][j] * Interval_ReLU(result[k-1][j]).ReLU()
                tmp += self.I_bias[k][i]
                tmpResult.append(tmp)
            result.append(tmpResult)
        
        self.forwardResult = result.copy()

    def backward_ReLU(self, hiddenLayerNumber, outputIndex, inputIndex):
        result = interval(0)
        if hiddenLayerNumber == 1:
            for i in range(self.layer_sizes[hiddenLayerNumber]):
                result += self.I_weights[hiddenLayerNumber][outputIndex][i] * Interval_ReLU(self.forwardResult[hiddenLayerNumber-1][i]).dReLU() * self.I_weights[0][i][inputIndex]
        else:
            for i in range(self.layer_sizes[hiddenLayerNumber]):
                result += self.I_weights[hiddenLayerNumber][outputIndex][i] * Interval_ReLU(self.forwardResult[hiddenLayerNumber-1][i]).dReLU() * self.backward_ReLU(hiddenLayerNumber-1, i, inputIndex)
          
        return result

    def myInterval_ReLU_old(self, I_X):
        self.forward_ReLU(I_X)

        result = interval[0]
        for i in range(self.inputs_size):
            tmp = self.backward_ReLU(self.hiddenLayerNumber, self.outputIndex, i)
            result += abs(tmp)

        return result

    def myInterval_ReLU(self, I_X):

        self.get_gradient(I_X)

        result = interval[0]
        for i in range(self.inputs_size):
            tmp = self.gradent[self.hiddenLayerNumber-1][self.outputIndex][i]
            result += abs(tmp)

        return result

    def bisection_ReLU(self, boxes):
        # get bisection list
        new_X = []
        for I_X in boxes:
            tmp = getBisectionList(I_X)
            for t in tmp:
                new_X.append(t)
        
        # calculate interval
        low = []
        up = []
        for I_X in new_X:
            self.get_gradient(I_X)
            tmp_norm = self.myInterval_ReLU(I_X)
            tmp_extrema = tmp_norm.extrema
            if len(tmp_extrema) != 1:
                low.append(tmp_extrema[0][0])
                up.append(tmp_extrema[1][0])
            else:
                low.append(tmp_extrema[0][0])
                up.append(tmp_extrema[0][0])
    
        # drop useless boxes
        result = []
        low_max = max(low)
        for i in range(len(new_X)):
            if up[i] > low_max:
                result.append(new_X[i])
    
        return result, low_max, max(up)


    def getParameters(self):
        # get network parameters
        I_weights = []
        I_bias = []
        for name, param in self.network.named_parameters():
            if "weight" in name:
                tmp =  convert_2D_matrix_radius(param.data.numpy(), self.radius_weights)
                I_weights.append(tmp)
            
            if "bias" in name:
                tmp = convert_array_radius(param.data.numpy(), self.radius_bias)
                I_bias.append(tmp)
        
        self.I_weights = I_weights.copy()
        self.I_bias = I_bias.copy()

    def compute_ReLU(self):
        # compute 
        result = self.myInterval_ReLU(self.I_X)

        # if need bisection
        boxes = [self.I_X]
        if self.needBisect:
            tmp = result.extrema
            if len(tmp) == 1:
                gap = 0
            else:
                gap = tmp[1][0] - tmp[0][0]

            count = 0
            while (gap > self.minGap) and (count < self.maxIteration) and (len(boxes) < self.maxBoxes) and (self.countSame < self.maxSame):
                count += 1
                boxes, low_max, up_max = self.bisection_ReLU(boxes)
                result = interval[low_max, up_max]
                gap = up_max - low_max
                
                if low_max == self.previousLowMax:
                    self.countSame += 1
                else:
                    self.previousLowMax = low_max
                    self.countSame = 0

        
        self.boxes = boxes
        return result
    
    def compute_ReLU_cpp(self, index, isMultiLayers):
        if isMultiLayers:
            storeParametersMulti(self.network, self.input, index=index)
        else:
            hidden_neural_num = self.layer_sizes[1]
            storeParameters(self.network, self.input, index=index)
    
            # Set Conditions
            experIndex = index
            comparedToCLEVER = False
    
            get_interval_Lipschitz_CPP(str(experIndex), self.radius_inputs, self.inputs_size, hidden_neural_num, self.output_size, comparedToCLEVER, self.maxIteration, self.minGap, self.maxBoxes)


    #######################################################
    # For Sigmoid
    #######################################################

    def forward_Sigmoid(self, I_X):
        result = []
        for k in range(self.hiddenLayerNumber):
            tmpResult = []
            for i in range(self.layer_sizes[k+1]):
                tmp = interval(0)
                for j in range(self.layer_sizes[k]):
                    if k == 0:
                        tmp += self.I_weights[k][i][j] * I_X[j]
                    else:
                        tmp += self.I_weights[k][i][j] * Interval_Sigmoid(result[k-1][j]).Sigmoid()
                tmp += self.I_bias[k][i]
                tmpResult.append(tmp)
            result.append(tmpResult)
        
        self.forwardResult = result.copy()

    def backward_Sigmoid(self, hiddenLayerNumber, outputIndex, inputIndex):
        result = interval(0)
        if hiddenLayerNumber == 1:
            for i in range(self.layer_sizes[hiddenLayerNumber]):
                result += self.I_weights[hiddenLayerNumber][outputIndex][i] * Interval_Sigmoid(self.forwardResult[hiddenLayerNumber-1][i]).dSigmoid() * self.I_weights[0][i][inputIndex]
        else:
            for i in range(self.layer_sizes[hiddenLayerNumber]):
                result += self.I_weights[hiddenLayerNumber][outputIndex][i] * Interval_Sigmoid(self.forwardResult[hiddenLayerNumber-1][i]).dSigmoid() * self.backward_Sigmoid(hiddenLayerNumber-1, i, inputIndex)
          
        return result

    def myInterval_Sigmoid(self, I_X):
        self.forward_Sigmoid(I_X)

        result = interval[0]
        for i in range(self.inputs_size):
            tmp = self.backward_Sigmoid(self.hiddenLayerNumber, self.outputIndex, i)
            result += abs(tmp)

        return result

    def bisection_Sigmoid(self, boxes):
        # get bisection list
        new_X = []
        for I_X in boxes:
            tmp = getBisectionList(I_X)
            for t in tmp:
                new_X.append(t)
        
        # calculate interval
        low = []
        up = []
        for I_X in new_X:
            tmp_norm = self.myInterval_Sigmoid(I_X)
            tmp_extrema = tmp_norm.extrema
            if len(tmp_extrema) != 1:
                low.append(tmp_extrema[0][0])
                up.append(tmp_extrema[1][0])
            else:
                low.append(tmp_extrema[0][0])
                up.append(tmp_extrema[0][0])
    
        # drop useless boxes
        result = []
        low_max = max(low)
        for i in range(len(new_X)):
            if up[i] > low_max:
                result.append(new_X[i])
    
        return result, low_max, max(up)


    def compute_Sigmoid(self):
        # get network parameters
        I_weights = []
        I_bias = []
        for name, param in self.network.named_parameters():
            if "weight" in name:
                tmp =  convert_2D_matrix_radius(param.data.numpy(), self.radius_weights)
                I_weights.append(tmp)
            
            if "bias" in name:
                tmp = convert_array_radius(param.data.numpy(), self.radius_bias)
                I_bias.append(tmp)
        
        self.I_weights = I_weights.copy()
        self.I_bias = I_bias.copy()
        

        # compute result
        result = self.myInterval_Sigmoid(self.I_X)

        # if need bisection
        boxes = [self.I_X]
        if self.needBisect:
            tmp = result.extrema
            if len(tmp) == 1:
                gap = 0
            else:
                gap = tmp[1][0] - tmp[0][0]

            count = 0
            while (gap > self.minGap) and (count < self.maxIteration) and (len(boxes) < self.maxBoxes) and (self.countSame < self.maxSame):
                count += 1
                boxes, low_max, up_max = self.bisection_Sigmoid(boxes)
                result = interval[low_max, up_max]
                gap = up_max - low_max
                
                if low_max == self.previousLowMax:
                    self.countSame += 1
                else:
                    self.previousLowMax = low_max
                    self.countSame = 0

                print(low_max)
                
        self.boxes = boxes
        return result