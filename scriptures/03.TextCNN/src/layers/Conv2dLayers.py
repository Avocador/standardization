import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dLayers(nn.Module):
    def __init__(self, args):
        super(Conv2dLayers, self).__init__()
        self.filter_sizes = args.filter_sizes
        self.num_filters_total = args.num_filters * len(self.filter_sizes)
        self.sequence_length = args.sequence_length
        self.filter_list = nn.ModuleList(
            [nn.Conv2d(1, args.num_filters, (size, args.embedding_size)) for size in args.filter_sizes]
        )

    def forward(self, X):
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(X))
            mp = nn.MaxPool2d((self.sequence_length - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

