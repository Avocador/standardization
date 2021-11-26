import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.factory.factory_data_io import DataIOFactory
from src.factory.factory_model import ModelFactory



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-io', type=str, default='seq_bilstm', choices=['sentences', 'seq_data', 'seq_bilstm'])
    parser.add_argument('--data-path', type=str, default='data in data_io_textbilstm.read_data()', choices=['./data/sentences.txt', './data/seq_data.txt', 'data in data_io_textbilstm.read_data()'])
    parser.add_argument('--model', type=str, default='textbilstm', choices=['textrnn', 'textlstm', 'textbilstm'])
    parser.add_argument('--n-step', type=int, default=0, choices=[0, 2, 3], help='textrnn: n_step=2, textlstm: n_step=3, textbilstm: n_step=0')
    parser.add_argument('--n-hidden', type=int, default=5, choices=[5, 128], help='textrnn: n_hidden=5, textlstm: n_hidden=128, textbilstm: n_hidden=5')
    args = parser.parse_args()

    data_io = DataIOFactory.create(args)
    input_batch, target_batch = data_io.processing(args)
    if args.model == 'textrnn':
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


    elif args.model == 'textlstm':
        char_arr, word_dict, number_dict, n_class, seq_data = data_io.char_arr, data_io.word_dict, data_io.number_dict, data_io.n_class, data_io.seq_data

        model = ModelFactory.create(args, n_class)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1000):
            optimizer.zero_grad()

            output = model(input_batch)
            loss = criterion(output, target_batch)
            if (epoch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()

        inputs = [sen[:3] for sen in seq_data]

        predict = model(input_batch).data.max(1, keepdim=True)[1]
        print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])

    elif args.model == 'textbilstm':

        data_io = DataIOFactory.create(args)
        input_batch, target_batch = data_io.processing(args)

        sentence, word_dict, number_dict, n_class, max_len = data_io.sentence, data_io.word_dict, data_io.number_dict, data_io.n_class, data_io.max_len

        model = ModelFactory.create(args, n_class)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        for epoch in range(10000):
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()

        predict = model(input_batch).data.max(1, keepdim=True)[1]
        print(sentence)
        print([number_dict[n.item()] for n in predict.squeeze()])



