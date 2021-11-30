from src.utils.batch_utils import make_changeword_batch

class ChangeWord():
    def __init__(self, args):
        super(ChangeWord, self).__init__()
        self.data_path = args.data_path
        self.sentences = None
        self.char_arr = None
        self.num_dic = None
        self.n_class = None
        self.batch_size = None
        self.n_step = args.n_step

    def read_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentences = list()
            for i in range(len(lines)):
                line = list(map(str, lines[i].split()))
                sentences.append(line)
        self.sentences = sentences
        return sentences

    def processing(self):
        sentences = self.read_data()
        char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
        self.char_arr = char_arr
        num_dic = {n: i for i, n in enumerate(char_arr)}
        self.num_dic = num_dic
        n_class = len(num_dic)
        self.n_class = int(n_class)
        batch_size = len(sentences)
        self.batch_size = batch_size

        input_batch, output_batch, target_batch = make_changeword_batch(sentences, self.n_step, n_class, num_dic)
        return input_batch, output_batch, target_batch



