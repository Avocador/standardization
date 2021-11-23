import torch
from src.utils.data_processing import make_batch

class Sentences():
    def __init__(self):
        super(Sentences, self).__init__()
        self.word_dict = None
        self.number_dict = None
        self.n_class = None
        self.batch_size = None
        self.sentences = None


    def read_data(self, args):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentences = list()
            for i in range(len(lines)):
                sentences.append(str(lines[i]))
        self.sentences = sentences
        return sentences

    def processing(self, args):
        sentences = self.read_data(args)
        word_sequence = ' '.join(sentences).split()
        word_list = list(set(word_sequence))
        self.word_dict = {w: i for i, w in enumerate(word_list)}
        self.number_dict = {i: w for i, w in enumerate(word_list)}
        self.n_class = len(self.word_dict)
        self.batch_size = len(sentences)

        input_batch, target_batch = make_batch(sentences, self.word_dict, self.n_class)
        input_batch = torch.FloatTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch




