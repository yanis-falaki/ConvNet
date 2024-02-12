import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleA(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(DownsampleA, self).__init__()
        self.pad = output_channels - input_channels
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(x)
        padding = torch.zeros(x.shape[0], self.pad, x.shape[2], x.shape[3], device=x.device)
        padded_x = torch.cat((x, padding), 1)
        return padded_x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, identity_downsample = None):
        super(ResidualBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.identity_downsample = identity_downsample

        stride = 1
        # If we're doubling the output channels, then output features must halve
        if input_channels != output_channels:
            stride = 2

        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = torch.add(x, identity)
        x = F.relu(x)
        return x



class ResNet(nn.Module):
    def __init__(self, n):
        super(ResNet, self).__init__()
        # Convolutional output dimensions formula (in each depth slice): W_new = (W-F + 2P)/S + 1 where W=input_shape, F=kernel_shape, P=padding_amount, S=stride_amount

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.convlayers = self._make_layer(n)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 10)

    def _make_layer(self, n):
        layers = []

        for i in range(n):
            layers.append(ResidualBlock(16, 16))
        
        layers.append(ResidualBlock(16, 32, DownsampleA(16, 32)))
        for i in range(n - 1):
            layers.append(ResidualBlock(32, 32))
        
        layers.append(ResidualBlock(32, 64, DownsampleA(32, 64)))
        for i in range(n - 1):
            layers.append(ResidualBlock(64, 64))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.convlayers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x