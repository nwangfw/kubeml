from torchvision import datasets
import numpy as np

train_data = datasets.CIFAR100('../data', train=True, download=True)
validation_data = datasets.CIFAR100('../data', train=False, download=True)



cifar100_x_train = train_data.data
cifar100_y_train = train_data.targets
cifar100_x_test = validation_data.data
cifar100_y_test = validation_data.targets


np.save("../data/cifar100_x_train", cifar100_x_train)
np.save("../data/cifar100_y_train", cifar100_y_train)
np.save("../data/cifar100_x_test", cifar100_x_test)
np.save("../data/cifar100_y_test", cifar100_y_test)


##./kubeml dataset create --name cifar100   --traindata /home/ning/repo/kubeml/data/cifar100_x_train.npy   --trainlabels /home/ning/repo/kubeml/data/cifar100_y_train.npy  --testdata /home/ning/repo/kubeml/data/cifar100_x_test.npy  --testlabels /home/ning/repo/kubeml/data/cifar100_y_test.npy