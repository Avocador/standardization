import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerLinear(LayerBase):
    def __init__(self, args, n_class):
        super(LayerLinear, self).__init__()
        self.linear = nn.Linear(args.n_hidden, n_class, bias=True)

    def forward(self, X):
        model = self.linear(X)
        return model