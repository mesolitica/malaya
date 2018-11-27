import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .text_functions import STOPWORDS
from tqdm import tqdm
import numpy as np

BETA = 0.75
ETA = 0.4


class Model:
    def __init__(
        self,
        data,
        unigram_distribution,
        word_vectors,
        doc_weights_init,
        n_topics = 25,
        batch_size = 32,
        lambda_const = 1.0,
        learning_rate = 1e-3,
    ):

        vocab_size, embedding_dim = word_vectors.shape
        n_windows = len(data)
        n_documents = len(np.unique(data[:, 0]))

        self.DOC_INDEX = tf.placeholder(tf.int32, shape = [None])
        self.WORD_INDEX = tf.placeholder(tf.int32, shape = [None])
        self.TARGET = tf.placeholder(
            tf.int32, shape = [None, data.shape[1] - 2]
        )

        doc_ids = data[:, 0]
        unique_docs, counts = np.unique(doc_ids, return_counts = True)
        weights = np.zeros((len(unique_docs)), 'float32')
        for i, j in enumerate(unique_docs):
            weights[j] = 1.0 / np.log(counts[i])
        weights = tf.convert_to_tensor(weights, np.float32)

        unigram_distribution = unigram_distribution / (
            np.sum(unigram_distribution ** BETA) + 1e-5
        )
        unigram_distribution = tf.convert_to_tensor(
            unigram_distribution, np.float32
        )

        self.topic_vectors = tf.get_variable(
            'topic_vectors',
            shape = (n_topics, embedding_dim),
            initializer = tf.orthogonal_initializer(),
        )

        alpha = 1.0 / n_topics
        self.embedding_doc = tf.get_variable(
            'self.embedding_doc',
            shape = doc_weights_init.shape,
            initializer = tf.initializers.constant(doc_weights_init),
            trainable = True,
        )
        self.embedding_word = tf.get_variable(
            'self.embedding_word',
            shape = word_vectors.shape,
            initializer = tf.initializers.constant(word_vectors),
            trainable = False,
        )

        embedded_doc = tf.nn.embedding_lookup(
            self.embedding_doc, self.DOC_INDEX
        )
        embedded_word = tf.nn.embedding_lookup(
            self.embedding_word, self.WORD_INDEX
        )
        doc_probs = tf.nn.softmax(embedded_doc)
        unsqueezed_doc_probs = tf.expand_dims(doc_probs, 2)
        unsqueezed_topic_vectors = tf.expand_dims(self.topic_vectors, 0)
        doc_vectors = tf.reduce_sum(
            unsqueezed_doc_probs * unsqueezed_topic_vectors, 1
        )

        w = tf.nn.embedding_lookup(weights, self.DOC_INDEX)
        w = (w / (tf.reduce_sum(w) + 1e-5)) * batch_size

        dropout_doc = tf.nn.dropout(doc_vectors, 0.5)
        dropout_word = tf.nn.dropout(embedded_word, 0.5)
        context_vectors = dropout_doc + dropout_word
        targets = tf.nn.embedding_lookup(self.embedding_word, self.TARGET)

        unsqueezed_context = tf.expand_dims(context_vectors, 1)
        log_targets = tf.log(
            tf.nn.sigmoid(tf.reduce_sum(targets * unsqueezed_context, 2))
        )
        noise = tf.multinomial(
            [unigram_distribution],
            batch_size * (data.shape[1] - 2) * batch_size,
        )
        noise = tf.reshape(
            noise, [batch_size, (data.shape[1] - 2) * batch_size]
        )

        noise = tf.nn.embedding_lookup(self.embedding_word, noise)
        noise = tf.reshape(
            noise, [batch_size, data.shape[1] - 2, batch_size, embedding_dim]
        )

        unsqueezed_context = tf.expand_dims(
            tf.expand_dims(context_vectors, 1), 1
        )

        sum_log_sampled = tf.log(
            tf.nn.sigmoid(
                tf.negative(tf.reduce_sum(noise * unsqueezed_context, 3))
            )
        )
        sum_log_sampled = tf.reduce_sum(sum_log_sampled, 2)

        neg_loss = log_targets + sum_log_sampled

        self.negative_loss = tf.negative(
            tf.reduce_mean(w * tf.reduce_sum(neg_loss, 1))
        )
        self.dirichlet_loss = tf.reduce_mean(
            w * tf.reduce_sum(tf.nn.softmax(embedded_word), 1)
        )
        self.dirichlet_loss = self.dirichlet_loss * (
            lambda_const * (1.0 - alpha)
        )
        self.cost = self.negative_loss + self.dirichlet_loss
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(
            zip(grads, tvars)
        )


def get_windows(doc, hws = 2):
    inside = [
        (w, doc[(i - hws) : i] + doc[(i + 1) : (i + hws + 1)])
        for i, w in enumerate(doc[hws:-hws], hws)
    ]
    beginning = [
        (w, doc[:i] + doc[(i + 1) : (2 * hws + 1)])
        for i, w in enumerate(doc[:hws], 0)
    ]
    end = [
        (w, doc[-(2 * hws + 1) : i] + doc[(i + 1) :])
        for i, w in enumerate(doc[-hws:], len(doc) - hws)
    ]
    return inside + beginning + end


def process_data(corpus, word2idx, min_words = 5):
    texts = []
    for sentence in corpus:
        if len(sentence.split()) > min_words:
            texts.append(sentence)
    corpus = texts
    texts = ' '.join(corpus)
    words = texts.split()
    word2freq = Counter(words)
    word_counts = [count for _, count in word2freq.most_common()]
    tokenized_docs = [(i, doc.split()) for i, doc in enumerate(corpus)]
    _words = set(words)
    encoded_docs = [
        (i, [word2idx[t] if t in word2idx else 0 for t in doc])
        for i, doc in tokenized_docs
    ]
    word_counts = np.array(word_counts)
    word_counts / sum(word_counts)
    data = []
    for index, (_, doc) in enumerate(encoded_docs):
        windows = get_windows(doc)
        data += [[index, w[0]] + w[1] for w in windows]

    return np.array(data, dtype = 'int64'), word_counts


def generate_lda(corpus, n_topics, max_df = 0.95, min_df = 2, temperature = 7):
    tf_vectorizer = CountVectorizer(
        max_df = max_df, min_df = min_df, stop_words = STOPWORDS
    )
    tfreq = tf_vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(
        n_topics = n_topics,
        max_iter = 5,
        learning_method = 'online',
        learning_offset = 50.0,
        random_state = 0,
    ).fit(tfreq)
    corpus_lda = lda.transform(tfreq)
    doc_weights_init = np.copy(corpus_lda)
    doc_weights_init = np.log(doc_weights_init + 1e-4)
    doc_weights_init /= temperature
    return doc_weights_init


def train_lda2vec(
    data,
    unigram_distribution,
    word_vectors,
    doc_weights_init,
    n_topics,
    batch_size = 32,
    epoch = 10,
):
    sess = tf.InteractiveSession()
    model = Model(
        data,
        unigram_distribution,
        word_vectors,
        doc_weights_init,
        n_topics = n_topics,
        batch_size = batch_size,
    )
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        pbar = tqdm(
            range(0, (data.shape[0] // batch_size) * batch_size, batch_size),
            desc = 'train minibatch loop',
        )
        for k in pbar:
            batch_doc = data[k : k + batch_size, 0]
            batch_word = data[k : k + batch_size, 1]
            batch_target = data[k : k + batch_size, 2:]
            negative_loss, dirichlet_loss, loss, _ = sess.run(
                [
                    model.negative_loss,
                    model.dirichlet_loss,
                    model.cost,
                    model.optimizer,
                ],
                feed_dict = {
                    model.DOC_INDEX: batch_doc,
                    model.WORD_INDEX: batch_word,
                    model.TARGET: batch_target,
                },
            )
            pbar.set_postfix(
                cost = loss,
                negative_loss = negative_loss,
                dirichlet_loss = dirichlet_loss,
            )
    doc_weights = model.embedding_doc.eval()
    topic_vectors = model.topic_vectors.eval()
    resulted_word_vectors = model.embedding_word.eval()
    similarity = np.matmul(topic_vectors, resulted_word_vectors.T)
    return doc_weights, similarity
