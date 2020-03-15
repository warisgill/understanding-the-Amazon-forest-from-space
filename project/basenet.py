import torch
import torch.nn as nn

from functools import partial
# from dataclasses import dataclass
import torch.nn.functional as F
from collections import OrderedDict


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes, alpha):
        super(ThreeLayerConvNet, self).__init__()

        self.alpha = alpha
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2, 
                               bias=True)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, 
                               bias=True)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.fc = nn.Linear(channel_2*32*32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        scores = None
        alpha = self.alpha
        scores = self.fc(flatten(F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(x), 
                                                          alpha)), alpha)))
        return scores


def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10, alpha=1e-2)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]
# test_ThreeLayerConvNet()