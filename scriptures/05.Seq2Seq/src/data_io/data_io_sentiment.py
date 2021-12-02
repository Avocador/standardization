import numpy as np
import torch


class Sentiment():
    def __init__(self, args):
        super(Sentiment, self).__init__()
        self.data_path = args.data_path
        self.sentences = None
        self.labels = None
        self.word_dict = None
        self.vocab_size = None

    def read_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            sentences = f.read().splitlines()
        self.sentences = sentences
        return sentences

    def processing(self):
        sentences = self.read_data()
        labels = [1, 1, 1, 0, 0, 0]
        self.labels = labels

        word_seq = ' '.join(sentences).split()
        word_list = list(set(word_seq))
        word_dict = {w: i for i, w in enumerate(word_list)}
        self.word_dict = word_dict
        vocab_size = len(word_dict)
        self.vocab_size = vocab_size

        inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
        targets = torch.LongTensor([out for out in labels])
        return inputs, targets



