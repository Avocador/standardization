from src.models.textrnn import TextRNN
from src.models.textlstm import TextLSTM
from src.models.textbilstm import TextBiLSTM

class ModelFactory():
    @staticmethod
    def create(args, n_class):
        if args.model == 'textrnn':
            return TextRNN(args, n_class)
        elif args.model == 'textlstm':
            return TextLSTM(args, n_class)
        elif args.model == 'textbilstm':
            return TextBiLSTM(args, n_class)


