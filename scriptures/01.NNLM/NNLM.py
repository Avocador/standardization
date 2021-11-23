import argparse

import torch
import torch.nn as nn
import torch.optim as optim


class TestSample():
    def __init__(self):
        self.sentences = sentences
        super(TestSample, self).__init__()

    def precessing_data(self):
        word_list = ' '.join(sentences).split()
        word_list = list(set(word_list))
        word2idx = {w: i for i,w in enumerate(word_list)}
        idx2word = {i: w for i, w in enumerate(word_list)}
        return word2idx, idx2word

    def classnum(self):
        word2idx, idx2word = self.precessing_data()
        n_class = len(word2idx)
        return n_class

    def make_batch(self):
        input_batch = []
        target_batch = []
        word2idx, idx2word = self.precessing_data()
        for sen in self.sentences:
            word = sen.split()
            input = [word2idx[n] for n in word[:-1]]
            target = word2idx[word[-1]]

            input_batch.append(input)
            target_batch.append(target)
        return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        n_class = TestSample().classnum()
        self.C = nn.Embedding(n_class, args.m)
        self.H = nn.Linear(args.n_step * args.m, args.n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(args.n_hidden))
        self.U = nn.Linear(args.n_hidden, n_class, bias=False)
        self.W = nn.Linear(args.n_step * args.m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))


    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, args.n_step * args.m)
        tanh = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(tanh)
        return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-step', type=int, default=2)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    # parser.add_argument('--sentences', type=list, action='append', nargs='+', default=['i like dog', 'i love coffee', 'i hate milk'])
    args = parser.parse_args()

    sentences = ['i love u', 'i hate u', 'i fuck her', 'i watch tv']

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    testsampe = TestSample()
    input_batch, target_batch = testsampe.make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    word2idx, idx2word = testsampe.precessing_data()



    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:6f}'.format(loss))
        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]

    print([sen.split()[:2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])










