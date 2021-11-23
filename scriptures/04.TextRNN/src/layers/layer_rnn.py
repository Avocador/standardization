import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerRNN(LayerBase):
    def __init__(self, args, n_class):
        super(LayerRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=args.n_hidden)

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.rnn(X, hidden)
        outputs = outputs[-1]
        return outputs