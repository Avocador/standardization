import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerBiLSTM(LayerBase):
    def __init__(self, args, n_class):
        super(LayerBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(n_class, args.n_hidden, bidirectional=True)
        self.n_hidden = args.n_hidden

    def forward(self, X):
        input = X.transpose(0, 1)
        hidden_state = torch.zeros(1*2, len(X), self.n_hidden)
        cell_state = torch.zeros(1*2, len(X), self.n_hidden)

        outputs, (_, _) = self.bilstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        return outputs

