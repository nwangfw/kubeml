from torchvision import datasets
import numpy as np

train_data = datasets.MNIST('../data', train=True, download=True)
validation_data = datasets.MNIST('../data', train=False, download=True)



mnist_x_train = train_data.data
mnist_y_train = train_data.targets
mnist_x_test = validation_data.data
mnist_y_test = validation_data.targets

print(len(mnist_x_train), len(mnist_y_train), len(mnist_x_test), len(mnist_y_test))


np.save("../data/mnist_x_train", mnist_x_train)
np.save("../data/mnist_y_train", mnist_y_train)
np.save("../data/mnist_x_test", mnist_x_test)
np.save("../data/mnist_y_test", mnist_y_test)

#command
#./kubeml dataset create --name mnist   --traindata /home/ning/repo/kubeml/data/mnist_x_train.npy   --trainlabels /home/ning/repo/kubeml/data/mnist_y_train.npy  --testdata /home/ning/repo/kubeml/data/mnist_x_test.npy  --testlabels /home/ning/repo/kubeml/data/mnist_y_test.npy