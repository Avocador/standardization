from src.models.changewordseq import ChangeWordSeq
from src.models.translateseqatten import TranslateSeqAtten
from src.models.sentimentbilstmseqatten import SentimentBiLSTMSeqAtten

class ModelFactory():
    @staticmethod
    def create(args, n_class):
        if args.model == 'ChangeWordSeq':
            return ChangeWordSeq(args, n_class)
        elif args.model == 'TranslateSeqAtten':
            return TranslateSeqAtten(args, n_class)
        elif args.model == 'SentimentBiLSTMSeqAtten':
            return SentimentBiLSTMSeqAtten(args, n_class) # n_class = vocab_size