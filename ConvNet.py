import torch
import torch.nn as nn
import torch.nn.functional as F

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional output dimensions formula (in each depth slice): W_new = (W-F + 2P)/S + 1 where W=input_shape, F=kernel_shape, P=padding_amount, S=stride_amount

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120) # in_features pre-calculated, 16 depth slices, shape is 5x5 - flattened
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self, X, return_feature_maps=False):
        pre_activation_maps = []
        post_activation_maps = []
        pool_activation_maps = []

        # First convolutional layer
        X = self.conv1(X)
        pre_activation_maps.append(X)

        # Activation function on conv1 output
        X = F.relu(X)
        post_activation_maps.append(X)

        # Max pooling on relu output, effectively halving the shape (kernel size of 2, stride of 2)
        X = self.pool(X)
        pool_activation_maps.append(X)

        # Second convolutional layer
        X = self.conv2(X)
        pre_activation_maps.append(X)

        # Activation on conv2 output
        X = F.relu(X)
        post_activation_maps.append(X)

        # Max Pooling on relu output
        X = self.pool(X)
        pool_activation_maps.append(X)

        # Flatten so that it can enter fully connected layer
        X = X.view(-1, 16*5*5)

        # Regular fully connected graph
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)

        if return_feature_maps:
            return X, pre_activation_maps, post_activation_maps, pool_activation_maps
        return X