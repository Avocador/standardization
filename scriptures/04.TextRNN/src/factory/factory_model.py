from src.models.textrnn import TextRNN

class ModelFactory():
    @staticmethod
    def create(args, n_class):
        if args.model == 'textrnn':
            return TextRNN(args, n_class)


