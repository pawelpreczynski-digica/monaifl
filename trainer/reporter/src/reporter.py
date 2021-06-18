import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import sys

from pathlib import Path
home = str(Path.home())

print(home)
modelpath = os.path.join(home, "monaifl", "save","models","client")
modelName = 'MNIST-test.pth.tar'
#modelName = 'monai-test.pth.tar'
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
    # create model object 
    model = Net()
    w_local = []
    # load model from path
    if (os.path.exists(modelFile)):
        model.load_state_dict(torch.load(modelFile))
        # copy state_dictionary in a temporary variable
        w_local = model.state_dict()
    # return temporary variable
    else:
        print("This seems to be first round. Please train the local model now by running testMNIST.py")
    return (w_local)   
    

def setLocalParameters(recv_params):
    # create model object 
    model = Net()
    # deserialize new model parameters
    model.load_state_dict(torch.load(recv_params))
    # set optimizer but disable learnability
    optimizer = optim.SGD(model.parameters(), lr=0.00, momentum=0.0)
    # evaluate model to reset the dropouts and batchnorms
    model.eval()
    # print model_state disctionary
    #print(model.state_dict())
    print("Updating Local Model...")
    #print file path
    print(modelFile)
    # serialize and save model file
    torch.save(model.state_dict(), modelFile)
    

# def get_parameters(self):
#         return [val.cpu().numpy() for _, val in model.state_dict().items()]