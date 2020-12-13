import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

def load_data(args):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

    return trainloader, testloader

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cudnn.benchmark = True
        print('backend: GPU')
    else:   
        device = torch.device('cpu')
        print('backend: CPU')
    return device