import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import models

parser = argparse.ArgumentParser(description='pytorch cifar10')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

EPOCHS = 10
BATCH_SIZE = 32

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# for dynamic logging
len_t = len(trainloader) # data_size / batch_size
dc_t = int(math.log10(len_t))+1 # digits count of len_t

# Define model, loss function and optimizers
model = models.LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

            if i % 500 == 499:
                print(f'[epoch {epoch+1}/{EPOCHS}, {i+1:{dc_t}}/{len_t}], loss: {running_loss/len_t:.3f}')
                running_loss = 0.0

# Test
def test():
    model.eval()
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print(f'GroundTruth: {" ".join("%5s" % classes[labels[j]] for j in range(5))}')
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted:   {" ".join("%5s" % classes[predicted[j]] for j in range(5))}')

    with torch.no_grad():
        test_loss = 0.0
        total = 0
        correct = 0
        for i, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'Loss: {test_loss/len(testloader):.3f}, Acc: {correct*100/total}% ({correct}/{total})')

def main():
    train()
    test()

if __name__ == '__main__':
    main()