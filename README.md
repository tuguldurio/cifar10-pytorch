# cifar10-pytorch
cifar10 implementation with pytorch

## Usage
```
git clone https://github.com/tuguldurio/cifar10-pytorch.git

cd cifar10-pytorch
```

Training network example:
```
python main.py -m lenet -epochs 5 --download
```
arguments:
```
--model, -m         required        training model name
--lr                default=1e-3    learning rate
--epoch             default=10      number of epochs to train
--batch-size        default=64      batch size
--log-interval      default=10      batches to wait before logging
--download          default=False   to download data
```

## License
[MIT](https://choosealicense.com/licenses/mit/)