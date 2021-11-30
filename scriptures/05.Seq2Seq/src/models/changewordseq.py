from src.layers.layer_base import LayerBase
from src.layers.layer_rnn import LayerRNN
from src.layers.layer_linear import LayerLinear

class ChangeWordSeq(LayerBase):
    def __init__(self, args, n_class):
        super(ChangeWordSeq, self).__init__()
        self.n_hidden = args.n_hidden
        self.enc_cell = LayerRNN(n_class, self.n_hidden)
        self.dec_cell = LayerRNN(n_class, self.n_hidden)
        self.fc = LayerLinear(self.n_hidden, n_class)


    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs)
        return model

