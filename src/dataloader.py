import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

def convert_targets(targets):
    """Convert MNIST labels
    Args:
        targets (array): Labels

    Returns:
        new_targets (array): Labels
    """
    new_targets = targets.clone().detach()
    new_targets[targets % 2 == 0] = 1 # even
    new_targets[targets % 2 != 0] = 0 # odd

    # print(new_targets[new_targets == 0].size(), new_targets[new_targets == 1].size())
    # => torch.Size([30508]) torch.Size([29492])

    return new_targets


class DataLoader(object):
    def __init__(self):
        # https://pytorch.org/docs/stable/torchvision/datasets.html
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        

    def get_data(self, split_rate=0.2, seed=1234):
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        train_data.targets = convert_targets(train_data.targets)

        # split data into train & validation set
        targets = train_data.targets
        train_indices, val_indices = train_test_split(np.arange(len(targets)), test_size=split_rate, random_state=seed, shuffle=True, stratify=targets)
        train_set = TensorDataset(train_data.data[train_indices], train_data.targets[train_indices])
        val_set = TensorDataset(train_data.data[val_indices], train_data.targets[val_indices])

        return train_set, val_set


    def get_test_data(self):
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        test_data.targets = convert_targets(test_data.targets)

        return test_data


if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.get_data()