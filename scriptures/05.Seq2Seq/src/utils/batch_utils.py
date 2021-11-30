import numpy as np
import torch

def make_changeword_batch(seq_data, n_step, n_class, num_dic):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'p' * (n_step - len(seq[i]))
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


def make_changeword_test_batch(input_word, n_step, n_class, num_dic):

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dic[n] for n in input_w]
    output = [num_dic[n] for n in 'S' + 'P' * n_step]

    input_batch = torch.FloatTensor(np.eye(n_class)[input]).unsqueeze(0)
    output_batch = torch.FloatTensor(np.eye(n_class)[output]).unsqueeze(0)

    return input_batch, output_batch