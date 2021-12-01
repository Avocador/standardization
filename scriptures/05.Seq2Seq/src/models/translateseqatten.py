import torch
import torch.nn.functional as F
from src.layers.layer_base import LayerBase
from src.layers.layer_rnn import LayerRNN
from src.layers.layer_linear import LayerLinear

class TranslateSeqAtten(LayerBase):
    def __init__(self, args, n_class):
        super(TranslateSeqAtten, self).__init__()
        self.n_hidden = args.n_hidden
        self.n_class = n_class
        self.enc_cell = LayerRNN(n_class, self.n_hidden)
        self.dec_cell = LayerRNN(n_class, self.n_hidden)

        self.attn = LayerLinear(self.n_hidden, self.n_hidden)
        self.out = LayerLinear(self.n_hidden * 2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)
        dec_inputs = dec_inputs.transpose(0, 1)

        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        model = torch.empty([n_step, 1, self.n_class])

        for i in range(n_step):
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)
            trained_attn.append(attn_weights.squeeze().data.numpy())

            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            dec_output = dec_output.squeeze(0)
            context = context.squeeze(1)
            model[i] = self.out(torch.cat((dec_output, context), 1))

        return model.transpose(0, 1).squeeze(0), trained_attn


    def get_att_weight(self, dec_output, enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)

        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):
        score = self.attn(enc_output)
        return torch.dot(dec_output.view(-1), score.view(-1))



