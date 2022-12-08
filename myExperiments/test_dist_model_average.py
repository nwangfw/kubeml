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


from torch import optim
from datetime import datetime
from torch.autograd import Variable

torch.manual_seed(42) 

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




model = models.resnet18(pretrained= False)
loaded_model = models.resnet18(pretrained= False)

compare_models(model, loaded_model)     

parameter_length = []
parameter_shape = []

model_length = 0
param_grad = np.zeros((1))

for param in model.parameters():
    tmp_shape = 1
    parameter_shape.append(param.data.numpy().shape)
    for w in param.data.numpy().shape:
        tmp_shape *=w
    parameter_length.append(tmp_shape)
    param_grad = np.concatenate((param_grad,param.data.numpy().flatten()))
    model_length += tmp_shape
param_grad = np.delete(param_grad,0)

print("model_length = {}".format(param_grad.shape))

# pull latest model
pos = 0
for layer_index, param in enumerate(loaded_model.parameters()):
    # TODO: understand how Variable class work
    # param.data = Variable(torch.from_numpy(np.asarray(param_grad[pos:pos+parameter_length[layer_index]],dtype=np.float32).reshape(parameter_shape[layer_index])))
    param.data = torch.from_numpy(np.asarray(param_grad[pos:pos+parameter_length[layer_index]],dtype=np.float32).reshape(parameter_shape[layer_index]))

    pos += parameter_length[layer_index]


compare_models(model, loaded_model)     


# how to assign it to multiple workers