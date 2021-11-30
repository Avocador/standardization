import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerRNN(LayerBase):
    def __init__(self, n_class, n_hidden):
        super(LayerRNN, self).__init__()
        self.rnn = nn.RNN(n_class, n_hidden, dropout=0.5)

    def forward(self, X, hidden):
        outputs, states = self.rnn(X, hidden)
        return outputs, states
