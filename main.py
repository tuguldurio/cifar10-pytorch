import time
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
def train(args, model, trainloader, criterion, optimizer, epoch, device):
    epoch_loss = 0.0
    step_loss = 0.0

    for i, (inputs, targets) in enumerate(trainloader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        step_loss += loss.item()

        if i % 10 == 0:
            print('[epoch {}/{}, {}/{}], loss: {:.3f}'.format(
                epoch, args.epochs,
                i, len(trainloader),
                step_loss / 10
                ), end='\r')
            step_loss = 0.0
        
    epoch_loss = epoch_loss / len(trainloader.dataset)
    return epoch_loss

# Test
def test(args, model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        test_loss /= len(testloader)
        print('\nTest Loss: {:.3f}, Acc: {}% ({}/{})'.format(
            test_loss, correct*100/total, correct, total
        ))

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images, labels = images.to(device), labels.to(device)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print('GroundTruth: {}'.format(' '.join("%5s" % classes[labels[j]] for j in range(5))))
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted:   {}'.format(' '.join("%5s" % classes[predicted[j]] for j in range(5))))

def main():
    parser = argparse.ArgumentParser(description='pytorch cifar10')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--download', default=False, action='store_true', help='whether download data or not')
    args = parser.parse_args()
    
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=args.download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=args.download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=True, num_workers=2)

    # cuda or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('GPU')
    else:   
        device = torch.device('cpu')
        print('CPU')

    # Define model, loss function and optimizers
    # model = models.VGG('VGG19').to(device)
    model = models.LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    # epoch loop
    for epoch in range(1, args.epochs+1):
        start = time.time()
        loss = train(args, model, trainloader, criterion, optimizer, epoch, device)
        took = time.time() - start
        print('[epoch {}/{}, {l}/{l}], loss: {:.3f}, took: {:.2f}s'.format(epoch, args.epochs, loss, took, l=len(trainloader)))
    print('took: {:.2f}s'.format(time.time()-start))
    test(args, model, testloader, criterion, device)

if __name__ == '__main__':
    main()