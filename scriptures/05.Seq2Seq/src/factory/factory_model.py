from src.models.changewordseq import ChangeWordSeq
from src.models.translateseqatten import TranslateSeqAtten

class ModelFactory():
    @staticmethod
    def create(args, n_class):
        if args.model == 'ChangeWordSeq':
            return ChangeWordSeq(args, n_class)
        elif args.model == 'TranslateSeqAtten':
            return TranslateSeqAtten(args, n_class)