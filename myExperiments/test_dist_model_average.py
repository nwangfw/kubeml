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
import shutil
import time

from torch import optim
from datetime import datetime
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp

torch.manual_seed(42) 
path = os.getcwd()

# clear old folders
if os.path.exists(f"{path}/tmp"):
    shutil.rmtree(f'{path}/tmp/')

if os.path.exists(f"{path}/merged"):
    shutil.rmtree(f'{path}/merged/')


os.makedirs(f"{path}/tmp", exist_ok=True)
os.makedirs(f"{path}/merged", exist_ok=True)




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


def reduce_scatter_epoch(vector, num_workers, my_rank, postfix):
    # each work has a new model vector
        # step 1: save other chunks to the corresponding folders
        # step 2: wait for its corresponding chunk folder to have worker - 1 file
        # step 3: get the merged chunk and save it to the corresponding merged folder
    # the chunks are merged together to get the new model


    # vector is supposed to be a 1-d numpy array
    num_all_values = vector.size
    num_values_per_worker = num_all_values // num_workers
    residue = num_all_values % num_workers

    cur_epoch = int(postfix)

    my_offset = (num_values_per_worker * my_rank) + min(residue, my_rank)
    my_length = num_values_per_worker + (1 if my_rank < residue else 0)
    my_chunk = vector[my_offset: my_offset + my_length]


    print('Process: {}, length: {}, range[{}, {}]'.format(my_rank, my_length, my_offset, my_offset + my_length))

    # write partitioned vector to the shared memory, except the chunk charged by myself
    for i in range(num_workers):
        if i != my_rank:
            offset = (num_values_per_worker * i) + min(residue, i)
            length = num_values_per_worker + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            key = "{}_{}".format(i, my_rank)
            # format of key in tmp-bucket: chunkID_workerID_epoch

            if not os.path.exists(f"{path}/tmp/chunk_{i}"):
                # if the demo_folder directory is not present 
                # then create it.
                os.makedirs(f"{path}/tmp/chunk_{i}") 

            np.save(f"{path}/tmp/chunk_{i}/{key}_{postfix}", vector[offset: offset + length])
            print('Process: {}, chunk id: {}, name: {}, range[{}, {}]'.format(my_rank, i, key, offset, offset + my_length))

    # read and aggregate the corresponding chunk
    
    

    while True:
        if os.path.isdir(f"{path}/tmp/chunk_{my_rank}"):
            # read file in
            break
        time.sleep(0.1)

    # load chunks from my rank to update vector
    # TODO: something is wrong here, need further check, 
    num_files = 0
    while num_files < num_workers - 1:
        files = os.listdir(f"{path}/tmp/chunk_{my_rank}")
        print("Folder name chunk_{}, number of files: {}".format(my_rank, len(files)))

        # wait for file
        while True:
            files = os.listdir(f"{path}/tmp/chunk_{my_rank}")
            if len(files) > 0:
                # read file in
                break
            time.sleep(0.1)

        if len(files) > 0:
            for file in files:

                #print("Process {}, file name {}".format(my_rank, file))
                #file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
                #can directly save the 
                #print("Process: ", my_rank, "file name: ", file)
                key_splits = file.split("_")
                #print("Process: ", my_rank, "key_splits[0]: ", key_splits[0], "key_splits[2]: ", key_splits[2])

                # if it's the chunk I care and it is from the current step
                # format of key in tmp-bucket: chunkID_workerID_epoch_batch
                if key_splits[0] == str(my_rank) and key_splits[2] == str(cur_epoch) + '.npy':
                    data = np.load(f"{path}/tmp/chunk_{my_rank}/{file}", allow_pickle=True)
                    print("File name: {}, File Type {}, File length: {}".format(file, type(data), len(data)))
                    print('data: {}, size: {}'.format(my_rank, len(data)))
                    my_chunk = my_chunk + data
                    num_files += 1
                    # The reason that we delete it is because that the order is not important
                    # While there is a file available, we calculate and then delete it 
                    #storage.delete(file_key, tmp_bucket)
                    # TODO: remember to uncomment it
                    os.remove(f"{path}/tmp/chunk_{my_rank}/{file}")
        
    # # write the aggregated chunk back
    # # key format in merged_bucket: chunkID_epoch_batch
    # #storage.save(my_chunk.tobytes(), str(my_rank) + '_' + postfix, merged_bucket)
    np.save(f"{path}/merged/merged_{str(my_rank)}_{postfix}", my_chunk)

    # read other aggregated chunks
    # merged_value = dict()
    # merged_value[my_rank] = my_chunk

    # num_merged_files = 0
    # already_read_files = []

    # TODO: remember to uncomment it
    # while num_merged_files < num_workers - 1:
    #     objects = storage.list(merged_bucket)

    #     if objects is not None:
    #         for obj in objects:
    #             file_key = urllib.parse.unquote_plus(obj["Key"], encoding='utf-8')
    #             key_splits = file_key.split("_")

    #             # key format in merged_bucket: chunkID_epoch
    #             # if not file_key.startswith(str(my_rank)) and file_key not in already_read:
    #             if key_splits[0] != str(my_rank) and key_splits[1] == str(cur_epoch) \
    #                     and file_key not in already_read_files:

    #                 data = storage.load(file_key, merged_bucket).read()
    #                 bytes_data = np.frombuffer(data, dtype=vector.dtype)

    #                 merged_value[int(key_splits[0])] = bytes_data

    #                 already_read_files.append(file_key)
    #                 num_merged_files += 1

    # # reconstruct the whole vector
    # result = merged_value[0]
    # for k in range(1, num_workers):
    #     result = np.concatenate((result, merged_value[k]))

    # return result

def model_to_vector(model):
    model_length = 0
    param_grad = np.zeros((1))

    parameter_length = []
    parameter_shape = []

    for param in model.parameters():
        tmp_shape = 1
        parameter_shape.append(param.data.numpy().shape)
        for w in param.data.numpy().shape:
            tmp_shape *=w
        parameter_length.append(tmp_shape)
        param_grad = np.concatenate((param_grad,param.data.numpy().flatten()))
        model_length += tmp_shape
    param_grad = np.delete(param_grad,0)
    return param_grad, model_length, parameter_shape, parameter_length

def vector_to_model(loaded_model, param_grad, model_length, parameter_shape, parameter_length):
    pos = 0
    for layer_index, param in enumerate(loaded_model.parameters()):
        # TODO: understand how Variable class work
        # param.data = Variable(torch.from_numpy(np.asarray(param_grad[pos:pos+parameter_length[layer_index]],dtype=np.float32).reshape(parameter_shape[layer_index])))
        param.data = torch.from_numpy(np.asarray(param_grad[pos:pos+parameter_length[layer_index]],dtype=np.float32).reshape(parameter_shape[layer_index]))

        pos += parameter_length[layer_index]
    return loaded_model



# model = models.resnet18(pretrained= False)
# loaded_model = models.resnet18(pretrained= False)

# compare_models(model, loaded_model)     

# # TODO: save it to .npy format
# parameter_vector, model_length, parameter_shape, parameter_length = model_to_vector(model)
# loaded_model = vector_to_model(loaded_model, parameter_vector, model_length, parameter_shape, parameter_length)


#compare_models(model, loaded_model)     

def train(num_workers, i, epoch):
    loaded_model = models.resnet18(pretrained= False)
    loaded_model.load_state_dict(torch.load(f"{path}/resnet18.pt"))
    parameter_vector, model_length, parameter_shape, parameter_length = model_to_vector(loaded_model)
    #print("process", i, "model_length", parameter_vector.shape, "first 50 elements",  parameter_vector[:50])

    reduce_scatter_epoch(parameter_vector, num_workers, i, epoch)

    # the first node has some extra work
    if i == 0:
        num_files = 0

        # load chunks from my rank to update vector
        while num_files < num_workers:
            files = os.listdir(f"{path}/merged")
            num_files = len(files)
            if num_files == num_workers:
                # still update model in a centralized way
                new_model_vector =  np.zeros((1))
                for i in range(num_workers):
                    data = np.load(f"{path}/merged/merged_{i}_{epoch}.npy")
                    print('Merged {} Length is {}'.format(i, len(data)))
                    print("data type: ", type(data))
                    print("data type: ", data.shape)

                    new_model_vector = np.concatenate((new_model_vector, data), axis=None)
                new_model_vector = np.delete(new_model_vector,0)                
                new_model_vector = new_model_vector/num_workers

                #new_model = models.resnet18(pretrained= False)

        #print("Merged model length :", new_model_vector.shape, "First 50 elements", new_model_vector[:50])
        loaded_model = vector_to_model(loaded_model, new_model_vector, model_length, parameter_shape, parameter_length)
        torch.save(loaded_model.state_dict(), f"{path}/resnet18_new.pt")
# make sure the model can be done by multiple workers.
# Start from sequential work




if __name__ == "__main__":  # confirms that the code is under main function

    model = models.resnet18(pretrained= False)
    torch.save(model.state_dict(), f"{path}/resnet18.pt")
    num_workers = 5
    for i in range(num_workers):
        os.makedirs(f"{path}/tmp/chunk_{i}", exist_ok=True)

    torch.multiprocessing.set_start_method('spawn')#

    for epoch in range(1):
        processes = []
        epochStart = datetime.now()
        procs = []
        # create optimizer
        for i in range(num_workers):
            proc = mp.Process(target=train,  args=(num_workers, i, epoch))  # instantiating without any argument
            procs.append(proc)
            proc.start()



        # complete the processes
        for proc in procs:
            proc.join()
        
        for p in processes:
            p.terminate()



        # model conversion back
        #loaded_model = vector_to_model(loaded_model, parameter_vector, model_length, parameter_shape, parameter_length)
        print("Epoch Time :", datetime.now() - epochStart)
    
    loaded_model = models.resnet18(pretrained= False)
    loaded_model.load_state_dict(torch.load(f"{path}/resnet18_new.pt"))
    compare_models(model, loaded_model)     
    shutil.rmtree(f'{path}/merged/')
    shutil.rmtree(f'{path}/tmp/')
    os.remove(f'{path}/resnet18.pt')
    os.remove(f'{path}/resnet18_new.pt')
