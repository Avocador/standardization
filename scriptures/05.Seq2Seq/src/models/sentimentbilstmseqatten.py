import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.layer_base import LayerBase
from src.layers.layer_lstm import LayerLSTM
from src.layers.layer_linear import LayerLinear


class SentimentBiLSTMSeqAtten(LayerBase):
    def __init__(self, args, vocab_size):
        super(SentimentBiLSTMSeqAtten, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.n_hidden = args.n_hidden
        self.num_classes = args.num_classes
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = LayerLSTM(self.embedding_dim, self.n_hidden)
        self.out = LayerLinear(self.n_hidden * 2, self.num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)

        hidden_state = torch.zeros(1 * 2, len(X), self.n_hidden)
        cell_state = torch.zeros(1 * 2, len(X), self.n_hidden)

        output, final_hidden_state, final_cell_state = self.lstm(input, hidden_state, cell_state)
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention

