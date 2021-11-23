import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np







class TextCNNBase(nn.Module):
    def __init__(self, vocab_size):
        super(TextCNNBase, self).__init__()

        self.sequence_length = 3
        self.embedding_size = 2
        self.filter_size = [1, 2, 3]
        self.channel_size = 3
        self.total_channel_size = self.channel_size * len(self.filter_size)
        self.class_num = 2
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.filter_list = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.channel_size, kernel_size=(size, self.embedding_size)) for size in self.filter_size]
        )
        self.linear = nn.Linear(self.total_channel_size, self.class_num, bias=True)


    def forward(self, X):       # X = [batch_size, sequence_length]
        char_embedding = self.embedding(X)  # char_embedding = [batch_size, sequence_length, embedding_size]
        char_embedding = char_embedding.unsqueeze(1)    # char_embedding = [batch_size, channel_size, sequence_length, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(char_embedding))
            mp = nn.MaxPool2d((self.sequence_length - self.filter_size[i] + 1, 1))
            pooled = mp(h).permute((0, 3, 2, 1))    # pooled = [batch_size, output_height, output_wide, output_channel]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_size))
        h_pool_flat = torch.reshape(h_pool, [-1, self.total_channel_size * 1 * 1])  # [batch_size, feature_size]
        model = self.linear(h_pool_flat)
        return model

if __name__ == '__main__':
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNNBase(vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])  # To using Torch Softmax Loss function

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_text = 'i hate me'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")








