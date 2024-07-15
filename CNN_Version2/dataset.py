from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader, SubsetRandomSampler


def dataset():
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=Compose([ToTensor()])
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=Compose([ToTensor()])
    )

    traindataloader = DataLoader(training_data, batch_size=4, shuffle=False, sampler=SubsetRandomSampler(range(100)))
    testdataloader = DataLoader(test_data, batch_size=4, shuffle=False, sampler=SubsetRandomSampler(range(20)))
    return traindataloader, testdataloader
