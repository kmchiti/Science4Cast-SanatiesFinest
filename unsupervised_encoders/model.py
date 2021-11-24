import torch
from torch import nn


class ff_network(nn.Module):
    def __init__(self,
                 features_num=15):
        """
        Fully Connected layers
        """
        super(ff_network, self).__init__()

        self.semnet = nn.Sequential(  # very small network for tests
            nn.Linear(features_num, 100),  # 15 properties
            # nn.BatchNorm1d(100),
            nn.ReLU(),

            nn.Linear(100, 100),
            nn.ReLU(),

            # nn.Linear(100, 100),
            # # nn.BatchNorm1d(100),
            # nn.ReLU(),

            nn.Linear(100, 10),
            # nn.BatchNorm1d(10),
            nn.ReLU(),

            nn.Linear(10, 1))

    def forward(self, x):
        """
        Pass throught network
        """
        res = self.semnet(x)

        return res


def model_generator(args):

    features_num = 6
    if 'baseline' not in args.features:
        features_num += len(args.features.split('_'))

    features_num *= 3

    model = ff_network(features_num)

    return model