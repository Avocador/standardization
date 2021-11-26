from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_linear import LayerLinear
from src.layers.layer_base import LayerBase

class TextBiLSTM(LayerBase):
    def __init__(self, args, n_class):
        super(TextBiLSTM, self).__init__()
        self.bilstm = LayerBiLSTM(args, n_class)
        args.n_hidden *= 2
        self.linear = LayerLinear(args, n_class)
        args.n_hidden //= 2

    def forward(self, X):
        outputs = self.bilstm(X)
        model = self.linear(outputs)
        return model

