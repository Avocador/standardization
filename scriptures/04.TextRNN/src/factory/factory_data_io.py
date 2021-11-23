from src.data_io.data_io_textrnn import Sentences

class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'sentences':
            return Sentences()