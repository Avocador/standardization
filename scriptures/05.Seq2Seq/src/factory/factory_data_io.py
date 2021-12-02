from src.data_io.data_io_dataset import ChangeWord
from src.data_io.data_io_translate import Translate
from src.data_io.data_io_sentiment import Sentiment

class DataIOFactory():
    @staticmethod
    def create(args):
        if args.data_io == 'ChangeWord':
            return ChangeWord(args)
        elif args.data_io == 'Translate':
            return Translate(args)
        elif args.data_io == 'Sentiment':
            return Sentiment(args)
