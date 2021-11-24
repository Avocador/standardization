import numpy as np

def make_batch(sentences, word_dict, n_class):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

def make_batch_lstm(seq_data, word_dict, n_class):
    input_batch, target_batch = [], []
    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]]
        target = word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return input_batch, target_batch
