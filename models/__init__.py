from .lenet import LeNet
from .vgg import VGG
from .alexnet import AlexNet

class VGG11(VGG):
    def __init__(self):
        super().__init__('VGG11')

class VGG13(VGG):
    def __init__(self):
        super().__init__('VGG13')

class VGG16(VGG):
    def __init__(self):
        super().__init__('VGG16')

class VGG19(VGG):
    def __init__(self):
        super().__init__('VGG19')

names = {
    'lenet': LeNet,
    'vgg11': VGG11,
    'vgg13': VGG13,
    'vgg16': VGG16,
    'vgg19': VGG19,
}