import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.lenet import LeNet

EPOCHS = 5

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# Define model, loss function and optimizers
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
def train():
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'epoch [{epoch+1}/{EPOCHS}], loss: {running_loss/len(trainloader):.3f}')
        running_loss = 0.0

# Test
def test():
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % testset.classes[labels[j]] for j in range(4)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % testset.classes[predicted[j]]
                                for j in range(4)))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

if __name__ == '__main__':
    train()
    test()