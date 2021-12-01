from src.data_io.data_io_dataset import ChangeWord
from src.data_io.data_io_translate import Translate

class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'ChangeWord':
            return ChangeWord(args)
        elif args.data_io == 'Translate':
            return Translate(args)
