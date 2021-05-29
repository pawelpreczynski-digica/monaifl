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


def getLocalParameters():
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.00, momentum=0.0)
    w_local = model.state_dict() # [val for key, val in model.state_dict().items()]
 #   print(w_local)
    return (w_local)
    
getLocalParameters()


# def get_parameters(self):
#         return [val.cpu().numpy() for _, val in model.state_dict().items()]