import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

class DataProcessor(object):
    # https://pytorch.org/docs/stable/torchvision/datasets.html
    def __init__(self):
        pass
        

    def get_data(self, split_rate=0.2, seed=1234):
        train_data = datasets.MNIST(root='./data', train=True, download=True)
        train_data.data = DataProcessor.convert_images(train_data.data)
        train_data.targets = DataProcessor.convert_targets(train_data.targets)

        # split data into train & validation set
        targets = train_data.targets
        train_indices, val_indices = train_test_split(np.arange(len(targets)), test_size=split_rate, random_state=seed, shuffle=True, stratify=targets)
        train_set = TensorDataset(train_data.data[train_indices], train_data.targets[train_indices])
        val_set = TensorDataset(train_data.data[val_indices], train_data.targets[val_indices])

        return train_set, val_set


    def get_test_data(self):
        test_data = datasets.MNIST(root='./data', train=False, download=True)
        train_data.data = DataProcessor.convert_images(test_data.data)
        test_data.targets = DataProcessor.convert_targets(test_data.targets)

        return test_data
    

    @staticmethod
    def convert_images(imgs):
        new_imgs = imgs.to(torch.float) / 255.
        new_imgs = new_imgs.view(new_imgs.size()[0], -1)
        return new_imgs


    @staticmethod
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
        new_targets = new_targets.unsqueeze(1).to(torch.float)

        # print(new_targets[new_targets == 0].size(), new_targets[new_targets == 1].size())
        # => torch.Size([30508, 1]) torch.Size([29492, 1])

        return new_targets


if __name__ == '__main__':
    dataloader = DataProcessor()
    dataloader.get_data()