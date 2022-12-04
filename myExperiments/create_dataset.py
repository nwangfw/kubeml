from torchvision import datasets
import numpy as np

train_data = datasets.CIFAR10('../data', train=True, download=True)
validation_data = datasets.CIFAR10('../data', train=False, download=True)



cifar10_x_train = train_data.data
cifar10_y_train = train_data.targets
cifar10_x_test = validation_data.data
cifar10_y_test = validation_data.targets


np.save("../data/cifar10_x_train", cifar10_x_train)
np.save("../data/cifar10_y_train", cifar10_y_train)
np.save("../data/cifar10_x_test", cifar10_x_test)
np.save("../data/cifar10_y_test", cifar10_y_test)