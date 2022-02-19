import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms 
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

def plotConvergency(experimentIndex, Nb, Ns, cleverResults, intervalResultsLow, intervalResultsUp, name):
    x = np.array(Nb)
    y = np.array(Ns)
    X, Y = np.meshgrid(x, y)
    Z = np.array(cleverResults)
    ZZ = np.array(intervalResultsLow)
    ZZZ = np.array(intervalResultsUp)

    path = "Saved/convergency/E" + str(experimentIndex)
    if not os.path.exists(path):
        os.makedirs(path)
    # if not os.path.exists(path + "/Ns"):
    #     os.makedirs(path + "/Ns")
    if not os.path.exists(path + "/Nb"):
        os.makedirs(path + "/Nb")

    for i in range(len(Nb)):
        tmp = np.array([np.array(cleverResults)[i], np.array(intervalResultsLow)[i], np.array(intervalResultsUp)[i]]).astype(np.float64)
        normlized = (tmp-tmp.min()) / (tmp.max() - tmp.min())
        plt.plot(y, normlized[0], label='Clever')
        plt.plot(y, normlized[1], label='Interval low')
        plt.plot(y, normlized[2], label='Interval up')
        matplotlib.rc('xtick', labelsize=12) 
        matplotlib.rc('ytick', labelsize=12)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(7, 3.5)
        # plt.xlabel('Number of Samples')
        # plt.ylabel('Normalized Lipschitz Constant')
        plt.savefig(path + "/Nb/Nb" + str(i) + ".png")
        plt.close()

    # for i in range(len(Ns)):
    #     tmp = np.array([np.array(cleverResults)[:, i], np.array(intervalResultsLow)[:, i], np.array(intervalResultsUp)[:, i]]).astype(np.float64)
    #     normlized = (tmp-tmp.min()) / (tmp.max() - tmp.min())
    #     plt.plot(y, normlized[0], label='Clever')
    #     plt.plot(y, normlized[1], label='Interval low')
    #     matplotlib.rc('xtick', labelsize=12) 
    #     matplotlib.rc('ytick', labelsize=12)
    #     fig = matplotlib.pyplot.gcf()
    #     fig.set_size_inches(7, 3.5)
    #     # plt.xlabel('Number of Samples')
    #     # plt.ylabel('Normalized Lipschitz Constant')
    #     plt.savefig(path + "/Ns/Ns" + str(i) + ".png")
    #     plt.close()


    # Creating figyre
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    # Creating plot
    ax.plot_surface(X, Y, Z)
    ax.plot_surface(X, Y, ZZ)
    ax.plot_surface(X, Y, ZZZ)

    plt.savefig(path + "/" + name + ".png")
    plt.close()


def plotAll(results, radius_list, index):
    path = "Saved/WeightBias/E" + str(index) + "/ResultsImage"
    if not os.path.exists(path):
            os.makedirs(path)
    with open(path+"/widthResults.txt", "a") as text_file:
        # offset list
        operations = [1, 0, -1]
        for i in range(len(results)):
            # all combination
            for j in range(len(results[i])):
                x = radius_list
                offset_x = operations[j // 3]
                offset_y = operations[j % 3]
                up = []
                low = []
                mid = []
                width = []
                for tmp in results[i][j]:
                    p1 = tmp[0][0]
                    p2 = tmp[0][1]
                    low.append(p1)
                    up.append(p2)
                    mid.append((p1+p2)/2)
                    width.append(p2-p1)
                index2 = str(i) + str(j)
                plotWidths(x, width, offset_x, offset_y, index, index2)
                plotMidpoints(x, mid, offset_x, offset_y, index, index2)
                plotMidExtrema(x, low, up, offset_x, offset_y, index, index2)
                text_file.write(str(width) + "\n\n")



def plotWidths(x, y, offset_x, offset_y, index, index2):
    plt.plot(x, y, linewidth=2)
    title = index2 + " offset X-" + str(offset_x) + " Y-" + str(offset_y)
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(7, 4.2)
    plt.xlabel('Radius')
    plt.ylabel('Lipschitz Constant')
    # plt.title(title)

    # Saving as img
    path = path = "Saved/WeightBias/E" + str(index) + "/ResultsImage" + "/widths/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + title + ".png")
    plt.close()


def plotMidpoints(x, y, offset_x, offset_y, index, index2):
    plt.plot(x, y, linewidth=2)
    title = index2 + " offset X-" + str(offset_x) + " Y-" + str(offset_y)
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(7, 4.2)
    plt.xlabel('Radius')
    plt.ylabel('Lipschitz Constant')
    # plt.title(title)

    # Saving as img
    path = path = "Saved/WeightBias/E" + str(index) + "/ResultsImage" + "/midpoints/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + title + ".png")
    plt.close()

def plotMidExtrema(x, low, up, offset_x, offset_y, index, index2):
    plt.plot(x, low, linewidth=2)
    plt.plot(x, up, linewidth=2)
    title = index2 + " offset X-" + str(offset_x) + " Y-" + str(offset_y)
    matplotlib.rc('xtick', labelsize=12) 
    matplotlib.rc('ytick', labelsize=12)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(7, 4.2)
    plt.xlabel('Radius')
    plt.ylabel('Lipschitz Constant')
    # plt.title(title)

    # Saving as img
    path = path = "Saved/WeightBias/E" + str(index) + "/ResultsImage" + "/extrema/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + title + ".png")
    plt.close()


def loadData(index, randomSeed):
    # IRIS dataset -- inputs: 4d; outputs: 3d
    if index==0:
        print("Using IRIS dataset.")
        iris = load_iris()
        X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, shuffle=True, random_state=randomSeed)
        inputNum = 4
        outputNum = 3

    # Digits dataset -- inputs: 64d; outputs: 10d
    elif index == 1:
        print("Using 64d digits dataset.")
        digits = load_digits()
        X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2, shuffle=True)
        inputNum = 64
        outputNum = 10

    # MNIST dataset -- inputs: 28*28 = 784d; outputs: 2d
    elif index == 2:
        print("Using MNIST dataset.")
        data_train = load_mnist_data('train', digits=[1,7], batch_size=16, shuffle=True).dataset # Training data
        data_test = load_mnist_data('val', digits=[1,7], batch_size=16, shuffle=True).dataset # Validation data 
        
        X_train = data_train.data.numpy()
        X_train = X_train.reshape(X_train.shape[0],-1)

        X_test = data_test.data.numpy()
        X_test = X_test.reshape(X_test.shape[0],-1)

        Y_train = data_train.targets.numpy()
        Y_test = data_test.targets.numpy()

        inputNum = 28*28
        outputNum = 10

    # Diabetes -- inputs: 8d; outputs: 2d
    elif index == 3:
        data=pd.read_csv('data/diabetes.csv')
        X=data.drop(['Outcome'], axis=1)
        y=data['Outcome']
        X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.2, random_state=10)
        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()
        Y_train = Y_train.values.tolist()
        Y_test = Y_test.values.tolist()

        inputNum = 8
        outputNum = 2

    # Balance-scale -- inputs: 4d; outputs: 3d
    elif index == 4:
        data=pd.read_csv('data/balance-scale.csv')
        X=data.drop(['Class'], axis=1)
        Y=data['Class']
        y = []
        for i in Y:
            if i == 'L':
                y.append(0)
            elif i == 'B':
                y.append(1)
            else:
                y.append(2)
        X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.2, random_state=10)
        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()

        inputNum = 4
        outputNum = 3

    elif index == 5:
         # read the data
        data=pd.read_csv('data/Iris_modified.csv')
        X=data.drop(['Id', 'Species'], axis=1)
        y=data['Species']
        X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.2, random_state=10)
        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()
        Y_train = Y_train.values.tolist()
        Y_test = Y_test.values.tolist()

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.LongTensor(Y_train)
    Y_test = torch.LongTensor(Y_test )
    return X_train, X_test, Y_train, Y_test


def load_mnist_data(train_or_val, digits=None, batch_size=128, shuffle=False,
                    use_cuda=False, dataset_dir="data"):
    """ Builds the standard MNIST data loader object for training or evaluation
        of MNIST data
    ARGS:
        train_or_val: string - must be 'train' or 'val' for training or 
                               validation sets respectively 

    """
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda
    dataloader_constructor = {'batch_size': batch_size, 
                              'shuffle': shuffle, 
                              'num_workers': 4,
                              'pin_memory': use_cuda}
    transform_chain = transforms.ToTensor()
    if digits == None:
        mnist_dataset = datasets.MNIST(root=dataset_dir, 
                                       train=(train_or_val == 'train'), 
                                       download=True, transform=transform_chain)
    else:
        mnist_dataset = SubMNIST(root=dataset_dir, digits=digits,
                                 train=(train_or_val=='train'), 
                                 download=True, transform=transform_chain)

    return torch.utils.data.DataLoader(mnist_dataset, **dataloader_constructor)

class SubMNIST(datasets.MNIST):
    valid_digits = set(range(10))
    def __init__(self, root, digits, train=True, transform=None, 
                 target_transform=None, download=False):
        super(SubMNIST, self).__init__(root, transform=transform, 
                                       target_transform=target_transform)
        assert [digit in self.valid_digits for digit in digits] 
        assert digits == sorted(digits)
        target_map = {digit + 10: i for i, digit in enumerate(digits)}
        
        # --- remap targets to select out only the images we want 
        self.targets = self.targets + 10
        for digit, label in target_map.items():
            self.targets[self.targets== digit] = label

        # --- then select only indices with these new labels 
        self.data = self.data[self.targets < 10]
        self.targets = self.targets[self.targets < 10]

    @property 
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property 
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')