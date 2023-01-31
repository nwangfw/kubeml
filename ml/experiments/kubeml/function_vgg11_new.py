import logging
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from serverlessdl import KubeModel, KubeDataset
from torch.optim import Adam, SGD
from torchvision.models.vgg import vgg11


# Mean and std as well as main training ideas gotten from
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
class Cifar10Dataset(KubeDataset):
    def __init__(self):
        super(Cifar10Dataset, self).__init__("cifar10")

        # this are the ones for cifar100
        # self.transf = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        #                          (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        # ])

        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.val_transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return self.transf(x), y.astype('int64')

    def __len__(self):
        return len(self.data)


class KubeVGG(KubeModel):

    def __init__(self, network, dataset: Cifar10Dataset):
        super(KubeVGG, self).__init__(network, dataset, gpu=True)

    # seems that there is some issues
    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     adam = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #     return adam

    def configure_optimizers(self) -> torch.optim.Optimizer:
        sgd = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        return sgd

    def train(self, batch, batch_index) -> float:

        criterion = nn.CrossEntropyLoss()
        # get the targets and labels from the batch
        x, y = batch

        self.optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y) 

        loss.backward()
        self.optimizer.step()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return loss.item()

    def validate(self, batch, batch_index) -> Tuple[float, float]:





        criterion = nn.CrossEntropyLoss()
        x, y = batch
        output = self(x)
        _, predicted = torch.max(output.data, 1)
        test_loss = criterion(output, y).item()
        correct = predicted.eq(y).sum().item()
        accuracy = correct * 100 / self.batch_size

        return accuracy, test_loss

    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass


def main():
    vgg = vgg11()
    dataset = Cifar10Dataset()
    kubenet = KubeVGG(vgg, dataset)
    return kubenet.start()
