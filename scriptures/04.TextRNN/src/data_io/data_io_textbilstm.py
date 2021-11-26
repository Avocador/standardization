import torch
from src.utils.data_processing import make_batch_bilistm

class Seq_bilstm():
    def __init__(self):
        super(Seq_bilstm, self).__init__()
        self.sentence = None
        self.word_dict = None
        self.number_dict = None
        self.n_class = None
        self.max_len = None

    def read_data(self, args):
        sentence = sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )
        self.sentence = sentence
        return sentence

    def processing(self, args):
        sentence = self.read_data(args)
        word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
        self.word_dict = word_dict
        number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
        self.number_dict = number_dict
        n_class = len(word_dict)
        self.n_class = n_class
        max_len = len(sentence.split())
        self.max_len = max_len

        input_batch, target_batch = make_batch_bilistm(sentence, word_dict, max_len, n_class)
        input_batch = torch.FloatTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch

