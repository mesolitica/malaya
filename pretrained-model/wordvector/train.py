import sys
import argparse
import pickle


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '%s is an invalid positive int value' % value
        )
    return ivalue


def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '%s is an invalid positive int value' % value
        )
    return ivalue


ap = argparse.ArgumentParser()
ap.add_argument('-t', '--text', required = True, help = 'text file to train')
ap.add_argument(
    '-e',
    '--embedding',
    required = True,
    type = check_positive,
    help = 'embedding size to train',
)
ap.add_argument(
    '-b',
    '--batch',
    type = check_positive,
    default = 256,
    help = 'batch size, default is 128',
)
ap.add_argument(
    '-v',
    '--vocab',
    type = check_positive,
    default = 1000000,
    help = 'maximum vocab size, default is 1000000',
)
ap.add_argument(
    '-lr',
    '--learning_rate',
    type = check_positive_float,
    default = 0.01,
    help = 'learning rate, default is 0.01',
)
ap.add_argument(
    '-epoch',
    '--epoch',
    type = check_positive,
    default = 10,
    help = 'epoch size, default is 10',
)
args = vars(ap.parse_args())

import word2vec
import numpy as np
import tensorflow as tf
import json
import os
import re
from unidecode import unidecode

os.environ['CUDA_VISIBLE_DEVICES'] = ''

filename = 'word2vec-%s-%d.p' % (args['text'].split('.')[0], args['embedding'])
print('save model to %s' % (filename))

batch_size = args['batch']
graph_params = {
    'batch_size': batch_size,
    'embed_size': args['embedding'],
    'hid_size': args['embedding'],
    'neg_samples': batch_size * 2,
    'learn_rate': args['learning_rate'],
    'momentum': 0.9,
    'embed_noise': 0.1,
    'hid_noise': 0.3,
    'epoch': args['epoch'],
    'optimizer': 'Momentum',
}

with open(args['text']) as fopen:
    sentences = fopen.read()

sentences = sentences.split()

word_array, dictionary, rev_dictionary, num_lines, num_words = word2vec.build_word_array(
    sentences, vocab_size = args['vocab']
)


X, Y = word2vec.build_training_set(word_array)

graph_params['vocab_size'] = np.max(X) + 1

split = round(X.shape[0] * 0.9)
train_X, train_Y = X[:split, :], Y[:split, :]
test_X, test_Y = X[split:, :], Y[split:, :]


model = word2vec.Model(graph_params)
print(
    'model built, vocab size %d, document length %d'
    % (np.max(X) + 1, len(word_array))
)

embed_weights, nce_weights = model.train(
    train_X,
    train_Y,
    test_X,
    test_Y,
    graph_params['epoch'],
    graph_params['batch_size'],
)

with open(filename, 'wb') as fopen:
    pickle.dump(
        {
            'dictionary': dictionary,
            'rev_dictionary': rev_dictionary,
            'embed_weights': embed_weights,
            'nce_weights': nce_weights,
        },
        fopen,
    )
