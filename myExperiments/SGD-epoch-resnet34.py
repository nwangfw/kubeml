# from common.experiment import KubemlExperiment, History, TrainOptions, TrainRequest
import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import torch.utils.data as tdata
from torch import optim
from datetime import datetime
from torch.nn.functional import nll_loss, cross_entropy

torch.manual_seed(42) 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
valset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256)
val_loader= torch.utils.data.DataLoader(valset, batch_size=256)


model = models.resnet.resnet34(pretrained= False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


def train(model: nn.Module, device,
          train_loader: tdata.DataLoader,
          optimizer: torch.optim.Optimizer, epoch) -> float:
    """Loop used to train the network"""

    # create optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)

    # load_state(optimizer)
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    loss, tot = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)


        loss = cross_entropy(output, target)
        tot += loss.item()
        

        loss.backward()
        optimizer.step()
        

        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
            

    # save the optimizer state
    # save_state(optimizer)

    return tot/len(train_loader)


def validate(model, device, val_loader: tdata.DataLoader) -> (float, float):
    """Loop used to validate the network"""

    criterion =nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_loss += cross_entropy(output, target).item()  # sum up batch loss
            correct += predicted.eq(target).sum().item()

    test_loss /= len(val_loader)

    accuracy = 100. * correct / len(val_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return accuracy, test_loss

def load_state(optimizer):
    if os.path.isfile('SGD-epoch-resnet34.pkl'):
        start = datetime.now()
        with open('SGD-epoch-resnet34.pkl', 'rb') as f:
            state = pickle.load(f)
            update_state(optimizer, state)
        print('loading optimizer state done, time is: ', datetime.now() - start)
    else:
        print('no state found')


def update_state(optimizer, state):
    state = {
      'param_groups': optimizer.state_dict()['param_groups'],
      'state': state
    }
    optimizer.load_state_dict(state)

def save_state(optimizer):
    start = datetime.now()
    with open('SGD-epoch-resnet34.pkl', 'wb') as f:
        pickle.dump(optimizer.state_dict()['state'], f)
    print('saving optimizer state done, time is: ', datetime.now() - start)


start = datetime.now()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

for epoch in range(10):
    print('\nEpoch', epoch)


    if epoch > 0:
        #load optimizer state from the previous epoch 
        load_state(optimizer)

    training_start = datetime.now()
    train(model, device, train_loader, optimizer, epoch)
    print('Training done, time is: ', datetime.now() - training_start)
    save_state(optimizer)
    validation_start = datetime.now()
    validate(model, device, val_loader)
    print('Validiation done, time is: ', datetime.now() - validation_start)



print("Training Completion Time: ", datetime.now() - start)
