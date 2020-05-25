
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
from unidecode import unidecode


def batch_sequence(sentences, dictionary, maxlen = 50):
    np_array = np.zeros((len(sentences), maxlen), dtype = np.int32)
    for no_sentence, sentence in enumerate(sentences):
        current_no = 0
        for no, word in enumerate(sentence.split()[: maxlen - 2]):
            np_array[no_sentence, no] = dictionary.get(word, 1)
            current_no = no
        np_array[no_sentence, current_no + 1] = 3
    return np_array


class Attention:
    def __init__(self,hidden_size):
        self.hidden_size = hidden_size
        self.dense_layer = tf.layers.Dense(hidden_size)
        self.v = tf.random_normal([hidden_size],mean=0,stddev=1/np.sqrt(hidden_size))
        
    def score(self, hidden_tensor, encoder_outputs):
        energy = tf.nn.tanh(self.dense_layer(tf.concat([hidden_tensor,encoder_outputs],2)))
        energy = tf.transpose(energy,[0,2,1])
        batch_size = tf.shape(encoder_outputs)[0]
        v = tf.expand_dims(tf.tile(tf.expand_dims(self.v,0),[batch_size,1]),1)
        energy = tf.matmul(v,energy)
        return tf.squeeze(energy,1)
    
    def __call__(self, hidden, encoder_outputs):
        seq_len = tf.shape(encoder_outputs)[1]
        batch_size = tf.shape(encoder_outputs)[0]
        H = tf.tile(tf.expand_dims(hidden, 1),[1,seq_len,1])
        attn_energies = self.score(H,encoder_outputs)
        return tf.expand_dims(tf.nn.softmax(attn_energies),1)

class Model:
    def __init__(
        self,
        dict_size,
        size_layers,
        learning_rate,
        maxlen,
        num_blocks = 3,
    ):
        block_size = size_layers
        self.BEFORE = tf.placeholder(tf.int32,[None,maxlen])
        self.INPUT = tf.placeholder(tf.int32,[None,maxlen])
        self.AFTER = tf.placeholder(tf.int32,[None,maxlen])
        self.batch_size = tf.shape(self.INPUT)[0]
        self.output_layer = tf.layers.Dense(dict_size, name="output_layer")
        self.output_layer.build(size_layers)
        self.embeddings = tf.Variable(tf.random_uniform([dict_size, size_layers], -1, 1))
        embedded = tf.nn.embedding_lookup(self.embeddings, self.INPUT)
        self.attention = Attention(size_layers)

        def residual_block(x, size, rate, block, reuse = False):
            with tf.variable_scope(
                'block_%d_%d' % (block, rate), reuse = reuse
            ):
                attn_weights = self.attention(tf.reduce_sum(x,axis=1), x)
                conv_filter = tf.layers.conv1d(
                    attn_weights,
                    x.shape[2] // 4,
                    kernel_size = size,
                    strides = 1,
                    padding = 'same',
                    dilation_rate = rate,
                    activation = tf.nn.tanh,
                )
                conv_gate = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size = size,
                    strides = 1,
                    padding = 'same',
                    dilation_rate = rate,
                    activation = tf.nn.sigmoid,
                )
                out = tf.multiply(conv_filter, conv_gate)
                out = tf.layers.conv1d(
                    out,
                    block_size,
                    kernel_size = 1,
                    strides = 1,
                    padding = 'same',
                    activation = tf.nn.tanh,
                )
                return tf.add(x, out), out

        forward = tf.layers.conv1d(
            embedded, block_size, kernel_size = 1, strides = 1, padding = 'SAME'
        )
        zeros = tf.zeros_like(forward)
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                forward, s = residual_block(
                    forward, size = 7, rate = r, block = i
                )
                zeros = tf.add(zeros, s)
        forward = tf.layers.conv1d(
            zeros,
            block_size,
            kernel_size = 1,
            strides = 1,
            padding = 'SAME',
            activation = tf.nn.tanh,
        )
        self.get_thought = tf.reduce_sum(forward,axis=1, name = 'logits')
        
        def decoder(labels, reuse):
            decoder_in = tf.nn.embedding_lookup(self.embeddings, labels)
            forward = tf.layers.conv1d(
                decoder_in, block_size, kernel_size = 1, strides = 1, padding = 'SAME'
            )
            zeros = tf.zeros_like(forward)
            for r in [8, 16, 24]:
                forward, s = residual_block(forward, size = 7, rate = r, block = 10, reuse = reuse)
                zeros = tf.add(zeros, s)
            return tf.layers.conv1d(
                zeros,
                block_size,
                kernel_size = 1,
                strides = 1,
                padding = 'SAME',
                activation = tf.nn.tanh,
            )
        
        fw_logits = decoder(self.AFTER, False)
        bw_logits = decoder(self.BEFORE, True)
        self.attention = tf.matmul(
            self.get_thought, tf.transpose(self.embeddings), name = 'attention'
        )
        self.loss = self.calculate_loss(fw_logits, self.AFTER) + self.calculate_loss(bw_logits, self.BEFORE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
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


def build_dict(word_counter, vocab_size = 200000):
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
    embedding_size = 64,
    maxlen = 50,
    **kwargs
):
    word_counter, _, _, _ = counter_words(train_X)
    dictionary, _ = build_dict(word_counter)
    print(len(dictionary))
    _graph = tf.Graph()
    with _graph.as_default():
        model = Model(
            len(dictionary),
            embedding_size,
            1e-3,
            maxlen,
        )
        sess = tf.InteractiveSession()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
    saver.save(sess, 'skip-wiki/model.ckpt')

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
    saver.save(sess, 'skip-wiki/model.ckpt')
    return sess, model, dictionary


# In[2]:


def cleaning(string):
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = ' '.join(
            [
                i
                for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string)
                if len(i)
            ]
        )
    return string.lower()

def split_by_dot(string):
    string = re.sub(
        r'(?<!\d)\.(?!\d)',
        'SPLITTT',
        string.replace('\n', '').replace('/', ' '),
    )
    string = string.split('SPLITTT')
    return [re.sub(r'[ ]+', ' ', sentence).strip() for sentence in string]

with open('wiki-ms.txt') as fopen:
    corpus = fopen.read()

print(corpus[:1000])
splitted = corpus.split()
corpus = []
for i in range(0, len(splitted), 50):
    corpus.append(' '.join(splitted[i:i+50]))
print(len(corpus))
corpus = corpus[100000:300000]
corpus = [cleaning(sentence) for sentence in corpus]

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
with open('skip-wiki-dict.json', 'w') as fopen:
    fopen.write(json.dumps(dictionary))