from src.layers.layer_base import LayerBase
from src.layers.layer_lstm import LayerLSTM
from src.layers.layer_linear import LayerLinear

class TextLSTM(LayerBase):
    def __init__(self, args, n_class):
        super(TextLSTM, self).__init__()
        self.layer_lstm = LayerLSTM(args, n_class)
        self.layer_linear = LayerLinear(args, n_class)


    def forward(self, X):
        outputs = self.layer_lstm(X)
        model = self.layer_linear(outputs)
        return model
