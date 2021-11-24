from src.data_io.data_io_textrnn import Sentences
from src.data_io.data_io_textlstm import Seq_data

class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'sentences':
            return Sentences()
        elif args.data_io == 'seq_data':
            return Seq_data()