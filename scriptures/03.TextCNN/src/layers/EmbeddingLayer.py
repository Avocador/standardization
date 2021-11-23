import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(args.voc_size, args.embedding_size)

    def forward(self, X):
        outputlayer = self.embedding(X)
        return outputlayer