import torch
from torch import nn


class ff_network(nn.Module):
    def __init__(self,
                 features_num=15,
                 hidden_layer=100):
        """
        Fully Connected layers
        """
        super(ff_network, self).__init__()

        self.semnet = nn.Sequential(  # very small network for tests
            nn.Linear(features_num, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.ReLU(),

            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),

            nn.Linear(hidden_layer, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),

            nn.Linear(10, 1))

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.semnet(x)

        return res
