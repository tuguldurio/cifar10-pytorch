import time
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import models

# Train
def train(args, model, trainloader, criterion, optimizer, epoch, device):
    model.train()
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

        if i % args.log_interval == 0:
            print('[epoch {}/{}, {}/{}], loss: {:.3f}'.format(
                epoch, args.epochs,
                i, len(trainloader),
                step_loss / args.log_interval
                ), end='\r')
            step_loss = 0.0
        
    epoch_loss = epoch_loss / len(trainloader.dataset)
    return epoch_loss

# Test
def test(args, model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            corrects += (predicted == targets).sum().item()
        test_loss /= len(testloader)
    return test_loss, corrects

def main():
    parser = argparse.ArgumentParser(description='pytorch cifar10')
    parser.add_argument('-m', '--model', type=str, choices=models.names.keys(), required=True, help='name of the model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--log-interval', default=10, type=int, 
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    # Load data
    train_loader, test_loader = utils.load_data(args)

    # cuda or cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('backend: GPU')
    else:   
        device = torch.device('cpu')
        print('backend: CPU')

    # Define model, loss function and optimizers
    model = models.names[args.model]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    # epoch loop
    for epoch in range(1, args.epochs+1):
        step_start = time.time()
        loss = train(args, model, trainloader, criterion, optimizer, epoch, device)
        test_loss, corrects = test(args, model, testloader, criterion, device)
        print('[epoch {}/{}, {l}/{l}], loss: {:.3f}, test_acc: {}%, took: {:.2f}s'.format(
            epoch, args.epochs,
            loss, corrects*100/len(testset),
            time.time()-step_start,
            l=len(trainloader)
            ))

    # entire time took to train
    print('took: {:.2f}s'.format(time.time()-start))

    # test
    test_loss, corrects = test(args, model, testloader, criterion, device)
    print('\nTest Loss: {:.3f}, Acc: {}% ({}/{})'.format(
            test_loss, corrects*100/len(testset), corrects, len(testset)
        ))

    # ground truth and prediction test
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('GroundTruth: {}'.format(' '.join("%5s" % classes[labels[j]] for j in range(5))))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   {}'.format(' '.join("%5s" % classes[predicted[j]] for j in range(5))))

if __name__ == '__main__':
    main()