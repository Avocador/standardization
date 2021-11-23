import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, args):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(args.num_filters * len(args.filter_sizes), args.num_classes, bias=False)

    def forward(self, X):
        outlayer = self.linear(X)
        return outlayer


