# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
import os
import numpy 
from pathlib import Path
home = str(Path.home())

print(home)
modelpath = os.path.join(home, "fl-architecture", "trainer", "save","models","client")

#ProjecttDir = os.getcwd()
#sys.path.insert(1, ProjecttDir)
#modelpath = './save/models'
 
modelName = 'MNIST-test.pth.tar'
modelFile = os.path.join(modelpath, modelName)

# Training settings
batch_size = 256

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


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


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)*100
    result = test_loss, test_accuracy
    return result
        #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))

if (os.path.exists(modelFile)):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    model.load_state_dict(torch.load(modelFile))
    model.eval()
#   model = torch.load(modelFile)

    for epoch in range(1, 3):
        train(epoch)
        loss, accuracy = test()
        print("Average Loss: " + str(loss.numpy()) + " Avergare Accuracy: "+ str(accuracy.numpy()))
 #  print(model.state_dict())

 
else:
    print("Local model does not exist...")
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, 3):
        train(epoch)
        loss, accuracy = test()
        print("Average Loss: " + str(loss.numpy()) + " Avergare Accuracy: "+ str(accuracy.numpy())) 

print(modelFile)
torch.save(model.state_dict(), modelFile)
