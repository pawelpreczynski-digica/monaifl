import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import sys

ProjecttDir = os.getcwd()
sys.path.insert(-2, ProjecttDir)


modelpath = '/save/models/client'
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


def getWeights():
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.00, momentum=0.0)
    #print(model.state_dict())
    #print(optimizer.state_dict())
    #model.load_state_dict(torch.load(modelFile))
    #model.eval()
    w_local = model.state_dict() # have to load the model
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])

    #return w_local

getWeights()