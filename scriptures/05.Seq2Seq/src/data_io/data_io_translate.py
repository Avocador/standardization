from src.utils.batch_utils import make_translate_batch

class Translate():
    def __init__(self, args):
        super(Translate, self).__init__()
        self.data_path = args.data_path
        self.sentences = None
        self.word_dict = None
        self.number_dict = None
        self.n_class = None

    def read_data(self):
        with open(self.data_path, 'r') as f:
            sentences = f.read().splitlines()
        self.sentences = sentences
        return sentences

    def processing(self):
        sentences = self.read_data()

        word_sequence = ' '.join(sentences).split()
        word_list = list(set(word_sequence))
        word_dict = {w: i for i, w in enumerate(word_list)}
        self.word_dict = word_dict
        number_dict = {i: w for i, w in enumerate(word_list)}
        self.number_dict = number_dict
        n_class = len(word_dict)
        self.n_class = n_class

        input_batch, output_batch, target_batch = make_translate_batch(sentences, word_dict, n_class)
        return input_batch, output_batch, target_batch





