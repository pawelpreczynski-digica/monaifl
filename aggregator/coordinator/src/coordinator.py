import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os.path
import sys

from pathlib import Path
home = str(Path.home())

print(home)
modelpath = os.path.join(home, "fl-architecture", "trainer", "save","models","server")

modelName = 'MNIST-test.pth.tar'
modelFile = os.path.join(modelpath, modelName)

#print (modelFile)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = np.divide(w_avg[k], len(w))
    return w_avg

def setGlobalParameters(recv_params):
    # create model object 
    model = Net()
    # deserialize new model parameters
    model.load_state_dict(torch.load(recv_params))
    # set optimizer but disable learnability
    #optimizer = optim.SGD(model.parameters(), lr=0.00, momentum=0.0)
    # evaluate model to reset the dropouts and batchnorms
    #model.eval()
    # print model_state disctionary
    print(model.state_dict())
    #print file path
    print(modelFile)
    # serialize and save model file
    torch.save(model.state_dict(), modelFile)
    