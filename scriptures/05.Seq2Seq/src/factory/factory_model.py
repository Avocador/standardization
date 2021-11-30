from src.models.changewordseq import ChangeWordSeq

class ModelFactory():
    @staticmethod
    def create(args, n_class):
        if args.model == 'ChangeWordSeq':
            return ChangeWordSeq(args, n_class)