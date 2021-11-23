import torch
import numpy as np

class DataIOFactory():
    @staticmethod
    def creat(args):
        if args.data_io == 'sentences':
            return DataIOSentences()

class DataIOSentences():
    def read_data(self, args):
        with open(args.sentences_path, 'r', encoding='utf-8') as f:
            sen_lines = f.readlines()
        with open(args.labels_path, 'r', encoding='utf-8') as l:
            lab_lines = l.readlines()
        sentences = list()
        labels = list()
        for i in range(len(sen_lines)):
            sen_line = str(sen_lines[i])
            lab_line = int(lab_lines[i])
            sentences.append(sen_line)
            labels.append(lab_line)
        return sentences, labels

    word2idx = None
    voc_size = None

    def process_input_target(self, args):
        sentences, labels = self.read_data(args)
        word_sequences = ' '.join(sentences).split()
        word_list = list(set(word_sequences))
        word2idx = {w: i for i, w in enumerate(word_list)}
        voc_size = len(word_list)
        self.word2idx = word2idx
        self.voc_size = voc_size

        inputs = torch.LongTensor([np.asarray([word2idx[n] for n in sen.split()]) for sen in sentences])
        targets = torch.LongTensor([out for out in labels])
        return inputs, targets




