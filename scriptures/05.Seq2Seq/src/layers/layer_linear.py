import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerLinear(LayerBase):
    def __init__(self, n_hidden, n_class):
        super(LayerLinear, self).__init__()
        self.linear = nn.Linear(n_hidden, n_class)

    def forward(self, X):
        outputs = self.linear(X)
        return outputs