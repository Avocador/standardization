import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_io.data_io_sentences import DataIOFactory
from src.models.TextCNN import TextCNN


parser = argparse.ArgumentParser()
parser.add_argument('--data-io', type=str, default='sentences')
parser.add_argument('--sentences-path', type=str, default='./data/sentences.txt')
parser.add_argument('--labels-path', type=str, default='./data/labels.txt')
parser.add_argument('--num-filters', type=int, default=3)
parser.add_argument('--filter_sizes', type=list, default=[2, 2, 2])
parser.add_argument('--embedding-size', type=int, default=2)
parser.add_argument('--sequence_length', type=int, default=3)
parser.add_argument('--num-classes', type=int, default=2)
args = parser.parse_args()



if __name__ == '__main__':

    data_io = DataIOFactory.creat(args)
    inputs, targets = data_io.process_input_target(args)
    args.voc_size = data_io.voc_size
    word2idx = data_io.word2idx
    model = TextCNN(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    print(word2idx)
    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word2idx[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, 'is Bad Mean...')
    else:
        print(test_text, 'is Good Mean!')






