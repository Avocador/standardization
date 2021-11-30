import argparse
import torch.nn as nn
import torch.optim as optim
from src.factory.factory_data_io import DataIOFactory
from src.factory.factory_model import ModelFactory
from src.utils.batch_utils import *




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-io', type=str, default='ChangeWord')
    parser.add_argument('--data-path', type=str, default='./data/changeword_sentences.txt')
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--model', type=str, default='ChangeWordSeq')
    parser.add_argument('--n-hidden', type=int, default=128)



    args = parser.parse_args()


    data_io = DataIOFactory.create(args)
    input_batch, output_batch, target_batch = data_io.processing()
    sentences, char_arr, num_dic, n_class, batch_size, n_step = data_io.sentences, data_io.char_arr, data_io.num_dic, data_io.n_class, data_io.batch_size, data_io.n_step

    model = ModelFactory.create(args, int(n_class))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):

        hidden = torch.zeros(1, batch_size, args.n_hidden)

        optimizer.zero_grad()

        output = model(input_batch, hidden, output_batch)

        output = output.transpose(0, 1)
        loss = 0
        for i in range(0, len(target_batch)):
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


    # Test
    def translate(word):
        input_batch, output_batch = make_changeword_test_batch(word, args.n_step, n_class, num_dic)

        hidden = torch.zeros(1, 1, args.n_hidden)
        output = model(input_batch, hidden, output_batch)

        predict = output.data.max(2, keepdim=True)[1]
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))


