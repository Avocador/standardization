import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.factory.factory_data_io import DataIOFactory
from src.factory.factory_model import ModelFactory



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-io', type=str, default='sentences')
    parser.add_argument('--data-path', type=str, default='./data/sentences.txt')
    parser.add_argument('--model', type=str, default='textrnn')
    parser.add_argument('--n-step', type=int, default=2)
    parser.add_argument('--n-hidden', type=int, default=5)
    args = parser.parse_args()

    data_io = DataIOFactory.create(args)
    input_batch, target_batch = data_io.processing(args)
    sentences, word_dict, number_dict, n_class, batch_size = data_io.sentences, data_io.word_dict, data_io.number_dict, data_io.n_class, data_io.batch_size

    model = ModelFactory.create(args, n_class)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        optimizer.zero_grad()

        hidden = torch.zeros(1, batch_size, args.n_hidden)
        output = model(hidden, input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]


    hidden = torch.zeros(1, batch_size, args.n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print(input, '->', [number_dict[n.item()] for n in predict.squeeze()])



