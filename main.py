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

# Train
def train(args, model, trainloader, criterion, optimizer):
    # for dynamic logging
    len_t = len(trainloader) # data_size / batch_size
    dc_t = int(math.log10(len_t))+1 # digits count of len_t

    for epoch in range(1, args.epochs+1):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader, 1):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 0:
                print('[epoch {}/{}, {:{}}/{}], loss: {:.3f}'.format(
                    epoch, args.epochs, i, dc_t, len_t, running_loss/len_t
                ), end='\r')
                running_loss = 0.0
        print()

# Test
def test(args, model, testloader, criterion):
    model.eval()
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('GroundTruth: {}'.format(' '.join("%5s" % classes[labels[j]] for j in range(5))))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   {}'.format(' '.join("%5s" % classes[predicted[j]] for j in range(5))))

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
        # print(f'Loss: {test_loss/len(testloader):.3f}, Acc: {correct*100/total}% ({correct}/{total})')
        print('Loss: {:.3f}, Acc: {}% ({}/{})'.format(
            test_loss/len(testloader), correct*100/total, correct, total
        ))

def main():
    parser = argparse.ArgumentParser(description='pytorch cifar10')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    args = parser.parse_args()
    
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=True, num_workers=2)

    # Define model, loss function and optimizers
    model = models.LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, trainloader, criterion, optimizer)
    test(args, model, testloader, criterion)

if __name__ == '__main__':
    main()