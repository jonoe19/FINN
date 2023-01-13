import torch as T
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dim) -> None:
        self.dim = (int(dim[0]/2), int(dim[0]/2))
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=33, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=97, out_features=1)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = T.cat((input, x), 1)
        x = F.max_pool2d(x, 2)
        max_pool = x
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = T.cat((max_pool, x), 1)
        x = F.avg_pool2d(x, self.dim)
        x = T.flatten(x, 1)
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x
