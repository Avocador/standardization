import torch
from src.utils.data_processing import make_batch_lstm

class Seq_data():

    def __init__(self):
        super(Seq_data, self).__init__()
        self.seq_data = None
        self.char_arr = None
        self.word_dict = None
        self.number_dict = None
        self.n_class = None

    def read_data(self, args):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            seq_data = list()
            for i in range(len(lines)):
                seq_data.append(str(lines[i][:-1]))
        self.seq_data = seq_data
        return seq_data

    def processing(self, args):
        seq_data = self.read_data(args)

        self.char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
        self.word_dict = {n: i for i, n in enumerate(self.char_arr)}
        self.number_dict = {i: w for i, w in enumerate(self.char_arr)}
        self.n_class = len(self.word_dict)


        input_batch, target_batch = make_batch_lstm(seq_data, self.word_dict, self.n_class)
        input_batch = torch.FloatTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        return input_batch, target_batch
