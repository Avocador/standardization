from src.layers.layer_base import LayerBase
import torch
import torch.nn as nn

class LayerLSTM(LayerBase):
    def __init__(self, args, n_class):
        super(LayerLSTM, self).__init__()
        self.lstm = nn.LSTM(n_class, args.n_hidden)
        self.n_hidden = args.n_hidden

    def forward(self, X, args):
        input = X.transpose(0, 1)

        hidden_state = torch.zeros(1, len(X), args.n_hidden)
        cell_state = torch.zeros(1, len(X), args.n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        return outputs
