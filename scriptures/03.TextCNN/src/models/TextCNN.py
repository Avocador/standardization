import torch
import torch.nn as nn
from src.layers.Conv2dLayers import Conv2dLayers
from src.layers.EmbeddingLayer import EmbeddingLayer
from src.layers.LinearLayer import LinearLayer

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.W = EmbeddingLayer(args)
        self.Weight = LinearLayer(args)
        self.Bias = nn.Parameter(torch.ones([args.num_classes]))
        self.con2vd = Conv2dLayers(args)

    def forward(self, X):
        embedded_chars = self.W(X)
        embedded_chars = embedded_chars.unsqueeze(1)
        h_pool_flat = self.con2vd(embedded_chars)
        model = self.Weight(h_pool_flat) + self.Bias
        return model


