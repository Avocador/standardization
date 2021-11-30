from src.data_io.data_io_dataset import ChangeWord


class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'ChangeWord':
            return ChangeWord(args)
