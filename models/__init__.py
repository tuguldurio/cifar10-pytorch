from .lenet import LeNet
from .vgg import VGG
from .alexnet import AlexNet
from .densenet import DenseNet, Bottleneck

# VGG
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


# Dense Net
class DenseNet121(DenseNet):
    def __init__(self):
        super().__init__(Bottleneck, [6, 12, 24, 16], growth_rate=32)

class DenseNet169(DenseNet):
    def __init__(self):
        super().__init__(Bottleneck, [6, 12, 32, 32], growth_rate=32)

class DenseNet201(DenseNet):
    def __init__(self):
        super().__init__(Bottleneck, [6, 12, 48, 32], growth_rate=32)

class DenseNet161(DenseNet):
    def __init__(self):
        super().__init__(Bottleneck, [6, 12, 36, 24], growth_rate=48)

class DenseNet121(DenseNet):
    def __init__(self):
        super().__init__(Bottleneck, [6, 12, 24, 16], growth_rate=12)

names = {
    'lenet': LeNet,
    'vgg11': VGG11,
    'vgg13': VGG13,
    'vgg16': VGG16,
    'vgg19': VGG19,
    'alexnet': AlexNet,
    'densenet121': DenseNet121,
}