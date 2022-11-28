import pymongo  # package for working with MongoDB
import uuid
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import pickle
from pymongo import collection
import pprint


client = pymongo.MongoClient("mongodb://localhost:27017/")


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
valset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
cifar10_x_train = trainset.data
cifar10_y_train = trainset.targets
cifar10_x_test = valset.data
cifar10_y_test = valset.targets

print(len(cifar10_x_train))
print(len(cifar10_y_train))
print(len(cifar10_x_test))
print(len(cifar10_y_test))

def dataset_splits(data, labels, batch_size):
    """ Given the data, return constantly sized
    batches of the dataset, which will be saved to the
    database"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]


def save_batches(col: collection.Collection, batches):
    """Saves the batches to the specified collection
    in the database"""
    ids = col.insert_many([
        {'_id': i,
         'data': pickle.dumps(data, pickle.HIGHEST_PROTOCOL),
         'labels': pickle.dumps(labels, pickle.HIGHEST_PROTOCOL)
         }
        for i, (data, labels) in enumerate(batches)
    ]).inserted_ids

def process_datasets(dataset_name):


    data, targets = None, None

    for datatype in ['train', 'test']:
        db = client[dataset_name]
        db.create_collection(datatype)

        splits = dataset_splits(data, targets, 64)
        save_batches(db[datatype], splits)



db = client['cifar10']
mycol = db["train"]


db_names = set(db.collection_names())
for dataset_name in ['train', 'test']:
    if dataset_name not in db_names:
        print(dataset_name)
        splits = dataset_splits(cifar10_x_train, cifar10_y_train, 64)
        save_batches(db[dataset_name], splits)

db_names = set(db.collection_names())

print(db_names)


# x = mycol.find_one()
result = mycol.find_one()

print ("The found_one() request returned ID:", result['_id'])
print ("The found_one() request returned ID:", len(pickle.loads(result['data'])))

print ("The found_one() request returned ID:", len(pickle.loads(result['labels'])))

#print ("The found_one() request returned ID:",  pickle.load(result["data"]))
#print ("The found_one() request returned ID:",  pickle.load(result["labels"]))
