from src.layers.layer_base import LayerBase
from src.layers.layer_rnn import LayerRNN
from src.layers.layer_linear import LayerLinear


class TextRNN(LayerBase):
    def __init__(self, args, n_class):
        super(TextRNN, self).__init__()
        self.rnn = LayerRNN(args, n_class)
        self.linear = LayerLinear(args, n_class)

    def forward(self, hidden, X):
        outputs = self.rnn(hidden, X)
        model = self.linear(outputs)
        return model


