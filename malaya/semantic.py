import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import re
import collections
import numpy as np
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from unidecode import unidecode
import itertools
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
data_index = 0

class VECTORIZE:
    vectors = None
    dictionary = None
    cosine = None
    euclidean = None
    keys = None

def clearstring(string):
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', '', string)
    string = word_tokenize(string)
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string).lower()
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))

def build_dataset(words, vocabulary_size):
    count = []
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        data.append(index)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, reverse_dictionary

def generate_batch_skipgram(words, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for i in range(span):
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    data_index = (data_index + len(words) - span) % len(words)
    return batch, labels

class Model:

    def __init__(self, batch_size, dimension_size, learning_rate, vocabulary_size):
        self.train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, dimension_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, dimension_size],
                                                           stddev = 1.0 / np.sqrt(dimension_size)))
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_weights, biases = self.nce_biases,
                                                  labels = self.train_labels,inputs=embed,
                                                  num_sampled = batch_size / 2, num_classes = vocabulary_size))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm

def train_vector(corpus, count_neighbors,iteration=10000,checkpoint=1000,dimension=64,
                 batch_size=128,skip_window=1,num_skip=2,learning_rate=1,cleaning=True):
    if cleaning:
        for i in range(len(corpus)): corpus[i] = clearstring(corpus[i])
    corpus = word_tokenize(' '.join(corpus))
    data, _, dictionary = build_dataset(corpus, len(set(corpus)))
    VECTORIZE.dictionary = dictionary
    print("Vocabulary size:", len(dictionary))
    tf.reset_default_graph()
    data_index = 0
    print("Creating Word2Vec model..")
    sess = tf.InteractiveSession()
    model = Model(batch_size, dimension, learning_rate, len(dictionary))
    sess.run(tf.global_variables_initializer())
    for step in range(iteration):
        batch_inputs, batch_labels = generate_batch_skipgram(data, batch_size, num_skip, skip_window)
        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}
        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
        if ((step + 1) % checkpoint) == 0:
            print("epoch: %d, loss: %f"%(step+1, loss))
    VECTORIZE.vectors = sess.run(model.normalized_embeddings)
    VECTORIZE.cosine = NearestNeighbors(count_neighbors, metric='cosine').fit(VECTORIZE.vectors)
    VECTORIZE.euclidean = NearestNeighbors(count_neighbors, metric='euclidean').fit(VECTORIZE.vectors)
    VECTORIZE.keys = list(VECTORIZE.dictionary.values())
    print("done train")

def semantic_search(word):
    ids = np.argmax([fuzz.ratio(i, word) for i in VECTORIZE.keys])
    xtest = VECTORIZE.vectors[ids, :].reshape((1,-1))
    distances, indices = VECTORIZE.cosine.kneighbors(xtest)
    out = []
    for no, i in enumerate(indices[0]):
        out.append((no, VECTORIZE.dictionary[i]))
    return out

def word_search(word):
    ids = np.argmax([fuzz.ratio(i, word) for i in VECTORIZE.keys])
    xtest = VECTORIZE.vectors[ids, :].reshape((1,-1))
    distances, indices = VECTORIZE.euclidean.kneighbors(xtest)
    out = []
    for no, i in enumerate(indices[0]):
        out.append((no, VECTORIZE.dictionary[i]))
    return out
