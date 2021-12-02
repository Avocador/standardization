import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.factory.factory_data_io import DataIOFactory
from src.factory.factory_model import ModelFactory
from src.utils.batch_utils import *




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-io', type=str, default='Sentiment', choices=['ChangeWord', 'Translate', 'Sentiment'])
    parser.add_argument('--data-path', type=str, default='./data/sentiment_sentences.txt', choices=['./data/changeword_sentences.txt', './data/translate_sentences.txt', './data/sentiment_sentences.txt'])
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--model', type=str, default='SentimentBiLSTMSeqAtten', choices=['ChangeWordSeq', 'TranslateSeqAtten', 'SentimentBiLSTMSeqAtten'])
    parser.add_argument('--n-hidden', type=int, default=5, choices=[5, 128], help='ChangeWord & Translate: 128, Sentiment: 5')
    parser.add_argument('--embedding-dim', type=int, default=2, help='Sentiment: 2')
    parser.add_argument('--num-classes', type=int, default=2, help='Sentiment: 2')



    args = parser.parse_args()


    data_io = DataIOFactory.create(args)
    if args.model == 'ChangeWordSeq':
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

    elif args.model == 'TranslateSeqAtten':
        input_batch, output_batch, target_batch = data_io.processing()
        sentences, word_dict, number_dict, n_class = data_io.sentences, data_io.word_dict, data_io.number_dict, data_io.n_class

        hidden = torch.zeros(1, 1, args.n_hidden)
        model = ModelFactory.create(args, n_class)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(2000):
            optimizer.zero_grad()
            output, _ = model(input_batch, hidden, output_batch)

            loss = criterion(output, target_batch.squeeze(0))
            if (epoch + 1) % 400 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()


        # Test
        test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
        test_batch = torch.FloatTensor(test_batch)
        predict, trained_att = model(input_batch, hidden, test_batch)
        predict = predict.data.max(1, keepdim=True)[1]
        print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    elif args.model == 'SentimentBiLSTMSeqAtten':
        inputs, targets = data_io.processing()
        sentences, labels, word_dict, vocab_size = data_io.sentences, data_io.labels, data_io.word_dict, data_io.vocab_size

        model = ModelFactory.create(args, vocab_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train
        for epoch in range(5000):
            optimizer.zero_grad()
            output, attention = model(inputs)
            loss = criterion(output, targets)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()

        # Test: Predict
        test_text = 'sorry hate you'
        tests = [np.asarray([word_dict[n] for n in test_text.split()])]
        test_batch = torch.LongTensor(tests)

        predict, _ = model(test_batch)
        predict = predict.data.max(1, keepdim=True)[1]
        if predict[0][0] == 0:
            print(test_text, 'is Bad Mean...')
        else:
            print(test_text, 'is Good Mean!')
