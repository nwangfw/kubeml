import multiprocessing as mp
import time
import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
import csv

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

import warnings
warnings.filterwarnings("ignore")
con = rai.Client(host="localhost", port=6379)


def assignProcssToDevice(nums_of_devices, id):
    return 'cuda:' + str(id % nums_of_devices)


def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')

def save_model(con, model, i):
    for key in model.state_dict().keys():
        redis_key = f'{i},{key}'
        #print(redis_key)
        con.tensorset(redis_key, model.state_dict()[key].cpu().numpy(), dtype='float')

def load_model(con, i):
    loaded_model = models.resnet34(pretrained= False)
    for key in loaded_model.state_dict().keys():
            #print(name)
        layer_weight = con.tensorget(f'{i},{key}')
        layer_weight_copied = np.copy(layer_weight)
        #print(type(classes))
        #counter += 1
        #print(classes.shape)
        
        loaded_model.state_dict()[key].copy_(torch.from_numpy(layer_weight_copied))
    return loaded_model

def train(device, epoch, i, results, args) -> float:
    """Loop used to train the network"""
    torch.manual_seed(42) 
    dataLoadTime = 0
    modelLoadTime = 0
    modelUpdateTime = 0
    modelSaveTime = 0
    trainEpochTime = 0

    trainStart = datetime.now()
    # create optimizer
    if epoch > 0:
        # load the global averaged model
        model = load_model(con, 'average')
    else:
        model = models.resnet34(pretrained= False)

 
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    if epoch > 0:
        load_state(optimizer, i)

    print('Process: {}, Time to load checkpoint: {}'.format(i, datetime.now() - trainStart))
    modelLoadTime = (datetime.now() - trainStart).total_seconds()

    criterion = nn.CrossEntropyLoss().to(device)
    
    dataLoadStart = datetime.now()
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=args.parallelism, rank=i)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    print('Process: {}  Data loading Time is {}'.format(i, datetime.now() - dataLoadStart)) 

    dataLoadTime = (datetime.now() - dataLoadStart).total_seconds()

    modelUpdateStart = datetime.now()
    model.train()
    loss, tot = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)


        loss = criterion(output, target)
        tot += loss.item()
        

        loss.backward()
        optimizer.step()
        

        # if batch_idx % 30 == 0:
        #     print('Process: {}, Device: {} Train Epoch: {} Step: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         i, device, epoch, batch_idx, len(train_loader.batch_sampler),
        #            100. * batch_idx / len(train_loader), loss.item()))
            
    print('Process: {}  Model Update Time is {}'.format(i, datetime.now() - modelUpdateStart)) 
    modelUpdateTime = (datetime.now() - modelUpdateStart).total_seconds()
    # save the optimizer state
    checkPointStart = datetime.now()
    save_state(optimizer, i)
    save_model(con, model, str(i))
    print('Process: {}, Time to save checkpoint: {}'.format(i, datetime.now() - checkPointStart))
    print('Process: {}, Time of one train epoch: {}'.format(i, datetime.now() - trainStart))
    modelSaveTime = (datetime.now() - checkPointStart).total_seconds()
    trainEpochTime = (datetime.now() - trainStart).total_seconds()

    if i == 0:
         # open the file in the write mode
        path = '{}/results.csv'.format(os.getcwd())
        with open(path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([dataLoadTime, modelLoadTime, modelUpdateTime, modelSaveTime, trainEpochTime])

    return tot/len(train_loader), dataLoadTime, modelLoadTime, modelUpdateTime, modelSaveTime, trainEpochTime


def validate(model, device) -> (float, float):
    """Loop used to validate the network"""

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    valset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    val_loader= torch.utils.data.DataLoader(valset, batch_size=args.batch_size)

    criterion =nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            #print(data.dtype, type(target))
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

def save_state(optimizer, i):
    with open(f'optimizer_state_{i}.pkl', 'wb') as f:
        #print(optimizer.state_dict()['state'])
        pickle.dump(optimizer.state_dict(), f)
    #print('saving optimizer state done, time is: ', datetime.now() - start)
def load_state(optimizer, i):
    if os.path.isfile(f'optimizer_state_{i}.pkl'):
        with open(f'optimizer_state_{i}.pkl', 'rb') as f:
            state = pickle.load(f)
            optimizer.load_state_dict(state)
            #update_state(optimizer, state)
        #print('loading optimizer state done, time is: ', datetime.now() - start)
    else:
        print('no state found')

def model_weight_average(parallelism):
    model = models.resnet34(pretrained = False)
    cur_model = models.resnet34(pretrained = False)

    sd_avg = model.state_dict()

    beta = 1.0/parallelism 
    for i in range(parallelism):
        cur_model = load_model(con, i)
        for key in cur_model.state_dict():
            if i == 0:
                sd_avg[key] = (cur_model.state_dict()[key]) / parallelism
            else:
                sd_avg[key] += (cur_model.state_dict()[key]) / parallelism
    model.load_state_dict(sd_avg)
    return model
    


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', default=128, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('-lr','--learning_rate', default=0.1, type=float, metavar='N',
                        help='learning rate')                    
    parser.add_argument('-p', '--parallelism', default=2, type=int,
                        help='number of functions')                        
    parser.add_argument('-e','--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()


    keys = ['data_loading', 'model_loading', 'model_update', 'model_saving', 'total']
    results = dict([(key, []) for key in keys])


    

    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=256)

    # May have to create our own dataloader
    # cifar10_x_train = trainset.data
    # cifar10_y_train = trainset.targets
    # cifar10_x_test = valset.data
    # cifar10_y_test = valset.targets

    # print(len(cifar10_x_train))
    # print(len(cifar10_y_train))
    # print(len(cifar10_x_test))
    # print(len(cifar10_y_test))


    nums_of_devices = torch.cuda.device_count()


    start_time = time.perf_counter()
    torch.multiprocessing.set_start_method('spawn')#

    start = datetime.now()
    parallelism = args.parallelism

    epoch_time = []
    model_averging_time = []


    for epoch in range(args.epochs):
        print('\nEpoch', epoch)
        processes = []
        epochStart = datetime.now()
        # create optimizer

        for i in range(parallelism):
            # not sure it will re-gernate the data or not.

            # print("Process ", i, dir(train_sampler), train_sampler.total_size)
            # print("Process ", i, dir(train_loader), type(train_loader.batch_sampler))
            # print("Process ", i, list(train_loader.batch_sampler))

            # for i, batch_indices in enumerate(train_loader.batch_sampler):
            #     print(f'Batch #{i} indices: ', batch_indices)
            device = assignProcssToDevice(nums_of_devices, i)
            # print(device)
            print('Process {}: Before Training Waiting time: {}'.format(i, datetime.now() - epochStart)) 

            p = mp.Process(target = train, args=(device, epoch, i, results, args))
            # p = multiprocessing.Process(target = task)

            p.start()
            processes.append(p)
   
        # Joins all the processes 
        for p in processes:
            p.join()
        
        processDeleteStart = datetime.now()
        for p in processes:
            p.terminate()
        print('Time to delete process : {}'.format(datetime.now() - processDeleteStart)) 

        modelAverageStart = datetime.now()
        model = model_weight_average(parallelism)
        save_model(con, model, 'average')
        print('Time to avergage and save models : {}'.format(datetime.now() - modelAverageStart)) 
        print('Time to finish one epoch : {}'.format(datetime.now() - epochStart)) 
        model_averging_time.append((datetime.now() - modelAverageStart).total_seconds())
        epoch_time.append((datetime.now() - start).total_seconds())






    print("Training Completion Time: ", (datetime.now() - start).total_seconds())
    finish_time = time.perf_counter()
    validation_start = datetime.now()
    validate(model, device)
    print('Validiation done, time is: ', datetime.now() - validation_start)
    print(f"Program finished in {finish_time-start_time} seconds")
    print(f"Epoch time in seconds")
    print(epoch_time)

    results = np.genfromtxt('{}/results.csv'.format(os.getcwd()), delimiter=',')
    print(['data_loading', 'model_loading', 'model_update', 'model_saving', 'total'])
    print(np.mean(results, axis=0))
    print(['model_average'])
    print(sum(model_averging_time) / float(len(model_averging_time)))
    os.remove('{}/results.csv'.format(os.getcwd()))