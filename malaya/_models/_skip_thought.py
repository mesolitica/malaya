import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import re
import collections
import json
import os
from .._utils._paths import PATH_SUMMARIZE, S3_PATH_SUMMARIZE
from .._utils._utils import download_file, load_graph


def batch_sequence(sentences, dictionary, maxlen = 50):
    np_array = np.zeros((len(sentences), maxlen), dtype = np.int32)
    for no_sentence, sentence in enumerate(sentences):
        current_no = 0
        for no, word in enumerate(sentence.split()[: maxlen - 2]):
            np_array[no_sentence, no] = dictionary.get(word, 1)
            current_no = no
        np_array[no_sentence, current_no + 1] = 3
    return np_array


class Model:
    def __init__(
        self,
        vocabulary_size,
        maxlen = 50,
        learning_rate = 1e-3,
        embedding_size = 256,
        **kwargs,
    ):
        word_embeddings = tf.Variable(
            tf.random_uniform(
                [vocabulary_size, embedding_size], -np.sqrt(3), np.sqrt(3)
            )
        )
        self.output_size = embedding_size
        self.maxlen = maxlen
        self.embeddings = word_embeddings
        self.output_layer = tf.layers.Dense(vocabulary_size)
        self.output_layer.build(self.output_size)

        self.BEFORE = tf.placeholder(tf.int32, [None, maxlen])
        self.INPUT = tf.placeholder(tf.int32, [None, maxlen])
        self.AFTER = tf.placeholder(tf.int32, [None, maxlen])
        self.batch_size = tf.shape(self.INPUT)[0]

        self.get_thought = self.thought(self.INPUT)
        self.attention = tf.matmul(
            self.get_thought, tf.transpose(self.embeddings), name = 'attention'
        )
        self.fw_logits = self.decoder(self.get_thought, self.AFTER)
        self.bw_logits = self.decoder(self.get_thought, self.BEFORE)
        self.loss = self.calculate_loss(
            self.fw_logits, self.AFTER
        ) + self.calculate_loss(self.bw_logits, self.BEFORE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss
        )

    def get_embedding(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def thought(self, inputs):
        encoder_in = self.get_embedding(inputs)
        fw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        sequence_length = tf.reduce_sum(tf.sign(inputs), axis = 1)
        with tf.variable_scope('thought_scope', reuse = False):
            rnn_output = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                encoder_in,
                sequence_length = sequence_length,
                dtype = tf.float32,
            )[1]
            return sum(rnn_output)

    def decoder(self, thought, labels):
        main = tf.strided_slice(labels, [0, 0], [self.batch_size, -1], [1, 1])
        shifted_labels = tf.concat([tf.fill([self.batch_size, 1], 2), main], 1)
        decoder_in = self.get_embedding(shifted_labels)
        cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        max_seq_lengths = tf.fill([self.batch_size], self.maxlen)
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_in, max_seq_lengths, time_major = False
        )
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, thought)
        decoder_out = tf.contrib.seq2seq.dynamic_decode(decoder)[0].rnn_output
        return decoder_out

    def calculate_loss(self, outputs, labels):
        mask = tf.cast(tf.sign(labels), tf.float32)
        logits = self.output_layer(outputs)
        return tf.contrib.seq2seq.sequence_loss(logits, labels, mask)


def counter_words(sentences):
    word_counter = collections.Counter()
    word_list = []
    num_lines, num_words = (0, 0)
    for i in sentences:
        words = re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', i)
        word_counter.update(words)
        word_list.extend(words)
        num_lines += 1
        num_words += len(words)
    return word_counter, word_list, num_lines, num_words


def build_dict(word_counter, vocab_size = 50000):
    count = [['PAD', 0], ['UNK', 1], ['START', 2], ['END', 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary, {word: idx for idx, word in dictionary.items()}


def news_load_model():
    if not os.path.isfile(PATH_SUMMARIZE['news']['model']):
        print('downloading SUMMARIZE news frozen model')
        download_file(
            S3_PATH_SUMMARIZE['news']['model'], PATH_SUMMARIZE['news']['model']
        )
    if not os.path.isfile(PATH_SUMMARIZE['news']['setting']):
        print('downloading SUMMARIZE news dictionary')
        download_file(
            S3_PATH_SUMMARIZE['news']['setting'],
            PATH_SUMMARIZE['news']['setting'],
        )
    g = load_graph(PATH_SUMMARIZE['news']['model'])
    x = g.get_tensor_by_name('import/Placeholder_1:0')
    logits = g.get_tensor_by_name('import/thought_scope/add_1:0')
    attention = g.get_tensor_by_name('import/attention:0')
    sess = tf.InteractiveSession(graph = g)
    with open(PATH_SUMMARIZE['news']['setting']) as fopen:
        dictionary = json.load(fopen)
    return sess, x, logits, attention, dictionary, 100


def wiki_load_model():
    if not os.path.isfile(PATH_SUMMARIZE['wiki']['model']):
        print('downloading SUMMARIZE wikipedia frozen model')
        download_file(
            S3_PATH_SUMMARIZE['wiki']['model'], PATH_SUMMARIZE['wiki']['model']
        )
    if not os.path.isfile(PATH_SUMMARIZE['wiki']['setting']):
        print('downloading SUMMARIZE wikipedia dictionary')
        download_file(
            S3_PATH_SUMMARIZE['wiki']['setting'],
            PATH_SUMMARIZE['wiki']['setting'],
        )
    g = load_graph(PATH_SUMMARIZE['wiki']['model'])
    x = g.get_tensor_by_name('import/Placeholder_1:0')
    logits = g.get_tensor_by_name('import/logits:0')
    attention = g.get_tensor_by_name('import/attention:0')
    sess = tf.InteractiveSession(graph = g)
    with open(PATH_SUMMARIZE['wiki']['setting']) as fopen:
        dictionary = json.load(fopen)
    return sess, x, logits, attention, dictionary, 50


def train_model(
    train_X,
    train_Y_before,
    train_Y_after,
    epoch = 10,
    batch_size = 16,
    embedding_size = 256,
    maxlen = 100,
    vocab_size = 50000,
    **kwargs,
):
    if not vocab_size:
        vocab_size = len(set(filter(None, (' '.join(train_X)).split()))) + 1
    word_counter, _, _, _ = counter_words(train_X)
    dictionary, _ = build_dict(word_counter, vocab_size = vocab_size)
    _graph = tf.Graph()
    with _graph.as_default():
        model = Model(
            len(dictionary),
            embedding_size = embedding_size,
            batch_size = batch_size,
            maxlen = maxlen,
            **kwargs,
        )
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

    for e in range(epoch):
        pbar = tqdm(range(0, len(train_X), batch_size), desc = 'minibatch loop')
        for i in pbar:
            batch_x = batch_sequence(
                train_X[i : min(i + batch_size, len(train_X))],
                dictionary,
                maxlen = maxlen,
            )
            batch_y_before = batch_sequence(
                train_Y_before[i : min(i + batch_size, len(train_X))],
                dictionary,
                maxlen = maxlen,
            )
            batch_y_after = batch_sequence(
                train_Y_after[i : min(i + batch_size, len(train_X))],
                dictionary,
                maxlen = maxlen,
            )
            loss, _ = sess.run(
                [model.loss, model.optimizer],
                feed_dict = {
                    model.BEFORE: batch_y_before,
                    model.INPUT: batch_x,
                    model.AFTER: batch_y_after,
                },
            )
            pbar.set_postfix(cost = loss)
    return sess, model, dictionary, saver


def load_skipthought(location, json):
    graph = tf.Graph()
    with graph.as_default():
        model = Model(
            len(json['dictionary']),
            embedding_size = json['embedding_size'],
            maxlen = json['maxlen'],
        )
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, location + '/model.ckpt')
    return sess, model, saver
