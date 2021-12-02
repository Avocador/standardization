import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerLSTM(LayerBase):
    def __init__(self, embedding_dim, n_hidden):
        super(LayerLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)

    def forward(self, input, hidden_state, cell_state):
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        return output, final_hidden_state, final_cell_state