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
from torch.nn.functional import nll_loss, cross_entropy
import redisai as rai

from torch import optim
from datetime import datetime

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

# model = models.resnet18(pretrained= False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


# model.to(device)

# def load_state(optimizer):
#     if os.path.isfile('standard_average_state2.pkl'):
#         with open('standard_average_state2.pkl', 'rb') as f:
#             state = pickle.load(f)
#             optimizer.load_state_dict(state)
#             #update_state(optimizer, state)
#         #print('loading optimizer state done, time is: ', datetime.now() - start)
#     else:
#         print('no state found')


# def update_state(optimizer, state):
#     state = {
#       'param_groups': optimizer.state_dict()['param_groups'],
#       'state': state
#     }
#     optimizer.load_state_dict(state)

# def save_state(optimizer):
#     with open('SGD-mini-batch-resnet18.pkl', 'wb') as f:
#         #print(optimizer.state_dict()['state'])
#         pickle.dump(optimizer.state_dict(), f)
#     #print('saving optimizer state done, time is: ', datetime.now() - start)
# def load_state(optimizer):
#     if os.path.isfile('SGD-mini-batch-resnet18.pkl'):
#         with open('SGD-mini-batch-resnet18.pkl', 'rb') as f:
#             state = pickle.load(f)
#             optimizer.load_state_dict(state)
#             #update_state(optimizer, state)
#         #print('loading optimizer state done, time is: ', datetime.now() - start)
#     else:
#         print('no state found')


# def update_state(optimizer, state):
#     state = {
#       'param_groups': optimizer.state_dict()['param_groups'],
#       'state': state
#     }
#     optimizer.load_state_dict(state)

# def save_state(optimizer):
#     with open('SGD-mini-batch-resnet18.pkl', 'wb') as f:
#         #print(optimizer.state_dict()['state'])
#         pickle.dump(optimizer.state_dict(), f)
#     #print('saving optimizer state done, time is: ', datetime.now() - start)

def validate(model, device, val_loader: tdata.DataLoader) -> (float, float):
    """Loop used to validate the network"""
    model.to(device)

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

# start = datetime.now()
# model.train()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# for epoch in range(10):
#     print('Epoch', epoch)

#     loss, tot = 0, 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()

#         output = model(data)
#         loss = cross_entropy(output, target)
#         tot += loss.item()
        

#         loss.backward()
#         optimizer.step()
        

#         if batch_idx % 30 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                    100. * batch_idx / len(train_loader), loss.item()))
            



#     validation_start = datetime.now()
#     validate(model, device, val_loader)
#     print('Validiation done, time is: ', datetime.now() - validation_start)



# print("Training Completion Time: ", datetime.now() - start)

# torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"checkpoint_resnet18.pt")


model = models.resnet18(pretrained= False)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

checkpoint = torch.load(f"checkpoint_resnet18.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



## load back and use validate function to test

# counter = -1
# for key in model.state_dict().keys():
#     print(key)
#     counter += 1
# print(counter)
# print("*"*50)
# for key in optimizer.state_dict().keys():
#     print(key)
# print("*"*50)





# print(dir(optimizer.state_dict()['state']))

# counter = -1
# for item in optimizer.state_dict()['state'].items():
#     print(item)
#     counter += 1
# print(counter)
# print(optimizer.state_dict()['state'][0]['momentum_buffer'])
# print(optimizer.state_dict()['state'][1]['momentum_buffer'])
# print(optimizer.state_dict()['state'][2]['momentum_buffer'])

# for value in optimizer.state_dict()['state'].values():
#     print(value['momentum_buffer'].shape)

# check model accuracy

validate(model, device, val_loader)
print("*"*50)

#start to connect to the redisAI
# documents can be found at https://redisai-py.readthedocs.io/en/stable/api.html
#print(model)
con = rai.Client(host="localhost", port=6379)
for key in model.state_dict().keys():

    con.tensorset(f'{key}',model.state_dict()[key].cpu().numpy(), dtype='float')

loaded_model = models.resnet18(pretrained= False)
validate(loaded_model, device, val_loader)

#counter = -1

# test case:
print("*"*100)
print(model.state_dict()['fc.bias'])
print((model.state_dict()['fc.bias']).dtype)

print("*"*100)
print(con.tensorget('fc.bias'))
print("*"*100)
print(loaded_model.state_dict()['fc.bias'])
print("before loading" + "*"*100)

for key in model.state_dict().keys():
        #print(name)
    layer_weight = con.tensorget(f'{key}')
    layer_weight_copied = np.copy(layer_weight)
    #print(type(classes))
    #counter += 1
    #print(classes.shape)
    
    loaded_model.state_dict()[key].copy_(torch.from_numpy(layer_weight_copied))
#print("redisAI counter:", counter)
print(loaded_model.state_dict()['fc.bias'])
print("After loading" +"*"*100)

loaded_model.to(device)
for key in model.state_dict().keys():
    print("*"*100)
    print(torch.sum(loaded_model.state_dict()[key] - model.state_dict()[key]))

validate(loaded_model, device, val_loader)

# test to see if the rebuilt model has the same performance 

# not sure how to save momentum into redis
# 
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


compare_models(model, loaded_model)       


# state_a = model.state_dict().__str__()
# state_b = loaded_model.state_dict().__str__()
 
# print(state_a)
# print("*"*100)
# print(state_b)
# if state_a == state_b:
#     print("Network not updating.")