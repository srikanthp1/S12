import torch
from torchvision import datasets, transforms

from .transforms import test_transforms, train_transforms


class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def get_dataset():
    train_data = Cifar10SearchDataset(
        root='./data/cifar10', train=True, download=True, transform=train_transforms)
    test_data = Cifar10SearchDataset(
        root='./data/cifar10', train=False, download=True, transform=test_transforms)

    return train_data, test_data