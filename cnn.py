import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet
from ConvNet import classes

writer = SummaryWriter('runs/experiment_1')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-paramters
num_epochs = 30
batch_size = 4
learning_rate = 0.001

# Dataset has PILImage images of range [0, 1]
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ConvNet().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(weight_decay=1e-4, params=model.parameters())

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    # to keep track of accumulated accuracy
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024 -- we get 4 from batch_size
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss

        # Calculate predictions
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)

        # Update epoch-wise counters
        epoch_correct += correct
        epoch_total += total

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.3f}, Acc: {(correct / total * 100):.3f} %')

        # Add step-wise loss, no accuracy since we're doing batches of 4 and so there's a lot of variability
        writer.add_scalar('step_loss', loss.item(), (epoch * n_total_steps) + i)

    print(f'Epoch [{epoch+1}/{num_epochs}], Accumulated_Loss: {(epoch_loss.item() / (n_total_steps)):.3f}, Accumulated_Acc: {(epoch_correct / epoch_total * 100):.3f} %')

    # Add epoch-wise loss and accuracy
    writer.add_scalar('accumulated_loss', epoch_loss.item() / (n_total_steps), epoch)
    writer.add_scalar('accumulated_accuracy', epoch_correct / epoch_total, epoch)

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # Max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]

torch.save(model.state_dict(), 'model_state_dict.pth')