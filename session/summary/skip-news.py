
# coding: utf-8

# In[1]:


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
from tensorflow.contrib import seq2seq


def sequence(s, w2v_model, maxlen = 50, vocabulary_size = 500000):
    words = s.split()
    np_array = np.zeros((maxlen), dtype = np.int32)
    current_no = 0
    for no, word in enumerate(words[: maxlen - 2]):
        id_to_append = 1
        if word in w2v_model:
            word_id = w2v_model[word]
            if word_id < vocabulary_size:
                id_to_append = word_id
        np_array[no] = id_to_append
        current_no = no
    np_array[current_no + 1] = 3
    return np_array


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
        output_size = 512,
        learning_rate = 1e-3,
        embedding_size = 256,
        batch_size = 16,
        max_grad_norm = 10,
        **kwargs
    ):
        word_embeddings = tf.Variable(
            tf.random_uniform(
                [vocabulary_size, embedding_size], -np.sqrt(3), np.sqrt(3)
            )
        )
        self.output_size = output_size
        self.maxlen = maxlen
        self.embeddings = word_embeddings
        self.output_layer = tf.layers.Dense(vocabulary_size)
        self.output_layer.build(output_size)

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
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), max_grad_norm
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def get_embedding(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def thought(self, inputs):
        encoder_in = self.get_embedding(inputs)
        fw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        sequence_length = tf.reduce_sum(tf.sign(inputs), axis = 1)
        with tf.variable_scope(
            'thought_scope', reuse = False
        ):
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
        helper = seq2seq.TrainingHelper(
            decoder_in, max_seq_lengths, time_major = False
        )
        decoder = seq2seq.BasicDecoder(cell, helper, thought)
        decoder_out = seq2seq.dynamic_decode(decoder)[0].rnn_output
        return decoder_out

    def calculate_loss(self, outputs, labels):
        mask = tf.cast(tf.sign(labels), tf.float32)
        logits = self.output_layer(outputs)
        return seq2seq.sequence_loss(logits, labels, mask)


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


def build_dict(word_counter, vocab_size = 500000):
    count = [['PAD', 0], ['UNK', 1], ['START', 2], ['END', 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary, {word: idx for idx, word in dictionary.items()}


def train_model(
    train_X,
    train_Y_before,
    train_Y_after,
    epoch = 10,
    batch_size = 16,
    embedding_size = 128,
    maxlen = 100,
    **kwargs
):
    word_counter, _, _, _ = counter_words(train_X)
    dictionary, _ = build_dict(word_counter)
    print(len(dictionary))
    _graph = tf.Graph()
    with _graph.as_default():
        model = Model(
            len(dictionary),
            embedding_size = embedding_size,
            output_size = embedding_size,
            batch_size = batch_size,
            maxlen = maxlen,
            **kwargs
        )
        sess = tf.InteractiveSession()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
    saver.save(sess, 'skip/model.ckpt')

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
    saver.save(sess, 'skip/model.ckpt')
    return sess, model, dictionary


# In[2]:


import json
with open('news-bm.json','r') as fopen:
    corpus = json.loads(fopen.read())

print(len(corpus))
corpus = [sentence for sentence in corpus if len(sentence) > 10]
print(len(corpus))


# In[3]:


stride = 1
t_range = int((len(corpus) - 3) / stride + 1)
left, middle, right = [], [], []
for i in range(t_range):
    slices = corpus[i * stride : i * stride + 3]
    left.append(slices[0])
    middle.append(slices[1])
    right.append(slices[2])


# In[5]:


len(left) == len(middle) == len(right)


# In[6]:


from sklearn.utils import shuffle
left, middle, right = shuffle(left, middle, right)


# In[ ]:


_,_,dictionary = train_model(middle,left,right)
with open('skip-news-dict.json', 'w') as fopen:
    fopen.write(json.dumps(dictionary))