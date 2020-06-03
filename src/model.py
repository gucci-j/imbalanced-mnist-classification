import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, drop_rate=0.4):
        super(Model, self).__init__()

        # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=drop_rate)

        self.dense2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.drop2 = nn.Dropout(p=drop_rate)

        self.dense3 = nn.Linear(int(hidden_dim / 2), 1)

    
    def forward(self, x):
        """
        Args:
            x: (batch_size, feature_dim)

        Returns:
            output: (batch_size, 1)
        """
        out1 = self.drop1(F.relu(self.dense1(x)))
        out2 = self.drop2(F.relu(self.dense2(out1)))
        output = self.dense3(out2)

        return output