import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import os
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from . import home
from ._utils._utils import download_file
from ._models import _word2vec
from .texts._text_functions import simple_textcleaning


def load_wiki():
    """
    Return malaya pretrained wikipedia word2vec size 256.

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    if not os.path.isfile('%s/word2vec-wiki/word2vec.p' % (home)):
        print('downloading word2vec-wiki embedded')
        download_file(
            'v13/word2vec/word2vec-wiki-nce-256.p',
            '%s/word2vec-wiki/word2vec.p' % (home),
        )
    with open('%s/word2vec-wiki/word2vec.p' % (home), 'rb') as fopen:
        return pickle.load(fopen)


def load_news(size = 256):
    """
    Return malaya pretrained news word2vec.

    Parameters
    ----------
    size: int, (default=256)

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    if not isinstance(size, int):
        raise ValueError('input must be an integer')
    if size not in [32, 64, 128, 256, 512]:
        raise ValueError(
            'size word2vec not supported, only supports [32, 64, 128, 256, 512]'
        )
    if not os.path.isfile('%s/word2vec-%d/word2vec.p' % (home, size)):
        print('downloading word2vec-%d embedded' % (size))
        download_file(
            'v7/word2vec/word2vec-%d.p' % (size),
            '%s/word2vec-%d/word2vec.p' % (home, size),
        )
    with open('%s/word2vec-%d/word2vec.p' % (home, size), 'rb') as fopen:
        return pickle.load(fopen)


def train(
    corpus,
    vocab_size = None,
    batch_size = 32,
    embedding_size = 256,
    hidden_size = 256,
    negative_samples_ratio = 0.5,
    learning_rate = 0.01,
    embedding_noise = 0.1,
    hidden_noise = 0.3,
    momentum = 0.9,
    epoch = 10,
    optimizer = 'momentum',
    text_cleaning = simple_textcleaning,
):
    """
    Train a word2vec for custom corpus.

    Parameters
    ----------
    corpus: list
        list of strings.
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus.
    embedding_size: int, (default=256)
        vector size representation for a word.
    hidden_size: int, (default=256)
        vector size representation for hidden layer.
    negative_samples_ratio: float, (default=0.5)
        negative samples ratio proportional to batch_size.
    learning_rate: float, (default=0.01)
        learning rate for word2vec.
    momentum: float, (default=0.9)
        momentum rate for optimizer=momentum.
    epoch: int, (default=5)
        iteration numbers.
    optimizer: str, (default='momentum')
        optimizer supported, ['gradientdescent', 'rmsprop', 'momentum', 'adagrad', 'adam']
    text_cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    if not isinstance(corpus, str) and not isinstance(corpus, list):
        raise ValueError('corpus must be a string or a list of string')
    if not vocab_size is None and not isinstance(vocab_size, int):
        raise ValueError('vocab_size must be a None or an integer')
    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer')
    if not isinstance(embedding_size, int):
        raise ValueError('embedding_size must be an integer')
    if not isinstance(hidden_size, int):
        raise ValueError('hidden_size must be an integer')
    if not isinstance(epoch, int):
        raise ValueError('epoch must be an integer')
    if not isinstance(negative_samples_ratio, float):
        raise ValueError('negative_samples_ratio must be a float')
    if not isinstance(momentum, float):
        raise ValueError('momentum must be a float')
    if not isinstance(embedding_noise, float):
        raise ValueError('embedding_noise must be a float')
    if not isinstance(hidden_noise, float):
        raise ValueError('hidden_noise must be a float')
    if not isinstance(learning_rate, float) or isinstance(learning_rate, int):
        raise ValueError('learning_rate must be a float or an integer')
    if not isinstance(optimizer, str):
        raise ValueError('optimizer must be a string')
    if not batch_size > 0:
        raise ValueError('batch_size must bigger than 0')
    if not epoch > 0:
        raise ValueError('epoch must bigger than 0')
    if not embedding_size > 0:
        raise ValueError('embedding_size must bigger than 0')
    if not hidden_size > 0:
        raise ValueError('hidden_size must bigger than 0')
    if not (negative_samples_ratio > 0 and negative_samples_ratio <= 1):
        raise ValueError(
            'negative_samples_ratio must bigger than 0 and less than or equal 1'
        )
    if not (embedding_noise > 0 and embedding_noise <= 1):
        raise ValueError(
            'embedding_noise must bigger than 0 and less than or equal 1'
        )
    if not (hidden_noise > 0 and hidden_noise <= 1):
        raise ValueError(
            'hidden_noise must bigger than 0 and less than or equal 1'
        )
    optimizer = optimizer.lower()
    if optimizer not in [
        'gradientdescent',
        'adam',
        'adagrad',
        'momentum',
        'rmsprop',
    ]:
        raise Exception(
            "Optimizer not supported, only supports ['gradientdescent', 'rmsprop', 'momentum', 'adagrad', 'adam']"
        )
    from sklearn.model_selection import train_test_split

    if isinstance(corpus, list):
        corpus = ' '.join(corpus)
    if text_cleaning:
        corpus = text_cleaning(corpus)
    corpus = ' '.join(corpus.split('\n'))
    corpus = list(filter(None, corpus.split()))
    if vocab_size is None:
        vocab_size = len(set(corpus)) + 5
    word_array, dictionary, rev_dictionary, num_lines, num_words = _word2vec.build_word_array(
        corpus, vocab_size
    )
    X, Y = _word2vec.build_training_set(word_array)
    graph_params = {
        'batch_size': batch_size,
        'vocab_size': np.max(X) + 1,
        'embed_size': embedding_size,
        'hid_size': hidden_size,
        'neg_samples': int(batch_size * negative_samples_ratio),
        'learn_rate': learning_rate,
        'momentum': momentum,
        'embed_noise': embedding_noise,
        'hid_noise': hidden_noise,
        'epoch': epoch,
        'optimizer': optimizer,
    }
    _, test_X, _, test_Y = train_test_split(X, Y, test_size = 0.1)
    model = _word2vec.Model(graph_params)
    print(
        'model built, vocab size %d, document length %d'
        % (np.max(X) + 1, len(word_array))
    )
    embed_weights, nce_weights = model.train(
        X, Y, test_X, test_Y, graph_params['epoch'], graph_params['batch_size']
    )
    return embed_weights, nce_weights, dictionary


class _Calculator:
    def __init__(self, tokens):
        self._tokens = tokens
        self._current = tokens[0]

    def exp(self):
        result = self.term()
        while self._current in ('+', '-'):
            if self._current == '+':
                self.next()
                result += self.term()
            if self._current == '-':
                self.next()
                result -= self.term()
        return result

    def factor(self):
        result = None
        if self._current[0].isdigit() or self._current[-1].isdigit():
            result = np.array([float(i) for i in self._current.split(',')])
            self.next()
        elif self._current is '(':
            self.next()
            result = self.exp()
            self.next()
        return result

    def next(self):
        self._tokens = self._tokens[1:]
        self._current = self._tokens[0] if len(self._tokens) > 0 else None

    def term(self):
        result = self.factor()
        while self._current in ('*', '/'):
            if self._current == '*':
                self.next()
                result *= self.term()
            if self._current == '/':
                self.next()
                result /= self.term()
        return result


class word2vec:
    def __init__(self, embed_matrix, dictionary):
        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}
        self.words = list(dictionary.keys())
        _graph = tf.Graph()
        with _graph.as_default():
            self._embedding = tf.placeholder(
                tf.float32, self._embed_matrix.shape
            )
            self._x = tf.placeholder(
                tf.float32, [None, self._embed_matrix.shape[1]]
            )
            normed_embedding = tf.nn.l2_normalize(self._embedding, axis = 1)
            normed_array = tf.nn.l2_normalize(self._x, axis = 1)
            self._cosine_similarity = tf.matmul(
                normed_array, tf.transpose(normed_embedding, [1, 0])
            )
            self._sess = tf.InteractiveSession()

    def get_vector_by_name(self, word):
        """
        get vector based on string.

        Parameters
        ----------
        word: str

        Returns
        -------
        vector: numpy
        """
        if not isinstance(word, str):
            raise ValueError('input must be a string')
        if word not in self._dictionary:
            arr = np.array([fuzz.ratio(word, k) for k in self.words])
            idx = (-arr).argsort()[:5]
            strings = ', '.join([self.words[i] for i in idx])
            raise Exception(
                'input not found in dictionary, here top-5 nearest words [%s]'
                % (strings)
            )
        return self._embed_matrix[self._dictionary[word]]

    def tree_plot(
        self,
        labels,
        notebook_mode = False,
        figsize = (7, 7),
        annotate = True,
        figname = 'fig.png',
    ):
        """
        plot a tree plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy.
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig.
        figsize : tuple, (default=(7, 7))
            figure size for plot.
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(annotate, bool):
            raise ValueError('annotate must be a boolean')
        if not isinstance(figname, str):
            raise ValueError('figname must be a string')
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )
        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        embed = self._embed_matrix[idx]
        embed = embed.dot(embed.T)
        embed = (embed - embed.min()) / (embed.max() - embed.min())
        labelled = []
        for label in labels:
            label = (
                '%s, %.3f' % (label[0], label[1])
                if isinstance(label, list)
                else label
            )
            labelled.append(label)
        plt.figure(figsize = figsize)
        g = sns.clustermap(
            embed,
            cmap = 'Blues',
            xticklabels = labelled,
            yticklabels = labelled,
            annot = annotate,
        )
        if notebook_mode:
            plt.show()
        else:
            plt.savefig(figname, bbox_inches = 'tight')

    def scatter_plot(
        self,
        labels,
        centre = None,
        notebook_mode = False,
        figsize = (7, 7),
        figname = 'fig.png',
        plus_minus = 25,
        handoff = 5e-5,
    ):
        """
        plot a scatter plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy
        centre : str, (default=None)
            centre label, if a str, it will annotate in a red color.
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig.
        figsize : tuple, (default=(7, 7))
            figure size for plot.
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(figname, str):
            raise ValueError('figname must be a string')
        if not isinstance(plus_minus, int):
            raise ValueError('plus_minus must be an integer')
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )
        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        if centre:
            if not isinstance(centre, str):
                raise ValueError('centre must be a string')
            idx.append(self.words.index(centre))
        cp_idx = idx[:]
        for i in idx:
            cp_idx.extend(np.arange(i - plus_minus, i).tolist())
            cp_idx.extend(np.arange(i, i + plus_minus).tolist())
        tsne = TSNE(n_components = 2, random_state = 0).fit_transform(
            self._embed_matrix[cp_idx]
        )
        plt.figure(figsize = figsize)
        plt.scatter(tsne[:, 0], tsne[:, 1])
        for label, x, y in zip(
            labels, tsne[: len(labels), 0], tsne[: len(labels), 1]
        ):
            label = (
                '%s, %.3f' % (label[0], label[1])
                if isinstance(label, list)
                else label
            )
            plt.annotate(
                label,
                xy = (x, y),
                xytext = (0, 0),
                textcoords = 'offset points',
            )
        if centre:
            plt.annotate(
                centre,
                xy = (tsne[len(labels), 0], tsne[len(labels), 1]),
                xytext = (0, 0),
                textcoords = 'offset points',
                color = 'red',
            )
        plt.xlim(
            tsne[: len(idx), 0].min() + handoff,
            tsne[: len(idx), 0].max() + handoff,
        )
        plt.ylim(
            tsne[: len(idx), 1].min() + handoff,
            tsne[: len(idx), 1].max() + handoff,
        )
        plt.xticks([])
        plt.yticks([])
        if notebook_mode:
            plt.show()
        else:
            plt.savefig(figname, bbox_inches = 'tight')

    def _calculate(self, equation):
        tokens, temp = [], ''
        for char in equation:
            if char == ' ':
                continue
            if char not in '()*+-':
                temp += char
            else:
                if len(temp):
                    row = self._dictionary[
                        self.words[
                            np.argmax([fuzz.ratio(temp, k) for k in self.words])
                        ]
                    ]
                    tokens.append(
                        ','.join(
                            self._embed_matrix[row, :].astype('str').tolist()
                        )
                    )
                    temp = ''
                tokens.append(char)
        if len(temp):
            row = self._dictionary[
                self.words[np.argmax([fuzz.ratio(temp, k) for k in self.words])]
            ]
            tokens.append(
                ','.join(self._embed_matrix[row, :].astype('str').tolist())
            )
        return _Calculator(tokens).exp()

    def _batch_process(self, batch, num_closest = 5, return_similarity = True):
        top_k = tf.nn.top_k(self._cosine_similarity, k = num_closest)
        results = self._sess.run(
            top_k,
            feed_dict = {self._x: batch, self._embedding: self._embed_matrix},
        )
        indices = results.indices
        values = results.values
        words = []
        if not return_similarity:
            for result in indices:
                words.append([self._reverse_dictionary[i] for i in result])
        else:
            for no in range(len(results)):
                words.append(
                    [
                        (
                            self._reverse_dictionary[indices[no, i]],
                            values[no, i],
                        )
                        for i in range(len(indices[no]))
                    ]
                )
        return words

    def batch_calculator(
        self, equations, num_closest = 5, return_similarity = False
    ):
        """
        batch calculator parser for word2vec using tensorflow.

        Parameters
        ----------
        equations: list of str
            Eg, '[(mahathir + najib) - rosmah]'
        num_closest: int, (default=5)
            number of words closest to the result.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(equations, list):
            raise ValueError('equations must be a list of string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(return_similarity, bool):
            raise ValueError('return_similarity must be a boolean')
        batches = np.array([self._calculate(eq) for eq in equations])
        return self._batch_process(
            batches,
            num_closest = num_closest,
            return_similarity = return_similarity,
        )

    def calculator(
        self,
        equation,
        num_closest = 5,
        metric = 'cosine',
        return_similarity = True,
    ):
        """
        calculator parser for word2vec.

        Parameters
        ----------
        equation: str
            Eg, '(mahathir + najib) - rosmah'
        num_closest: int, (default=5)
            number of words closest to the result.
        metric: str, (default='cosine')
            vector distance algorithm.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(equation, str):
            raise ValueError('input must be a string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(metric, str):
            raise ValueError('metric must be a string')
        if not isinstance(return_similarity, bool):
            raise ValueError('num_closest must be a boolean')
        calculated = self._calculate(equation)
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric = metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(calculated.reshape((1, -1)))
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            closest_indices = self.closest_row_indices(
                calculated, num_closest + 1, metric
            )
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            return word_list

    def batch_n_closest(
        self, words, num_closest = 5, return_similarity = False, soft = True
    ):
        """
        find nearest words based on a batch of words using Tensorflow.

        Parameters
        ----------
        words: list
            Eg, ['najib','anwar']
        num_closest: int, (default=5)
            number of words closest to the result.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.
        soft: bool, (default=True)
            if True, a word not in the dictionary will be replaced with nearest fuzzywuzzy ratio.
            if False, it will throw an exception if a word not in the dictionary.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(words, list):
            raise ValueError('input must be list of strings')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(return_similarity, bool):
            raise ValueError('return_similarity must be a boolean')
        if not isinstance(soft, bool):
            raise ValueError('soft must be a boolean')
        if soft:
            for i in range(len(words)):
                if words[i] not in self.words:
                    words[i] = self.words[
                        np.argmax([fuzz.ratio(words[i], k) for k in self.words])
                    ]
        else:
            for i in range(len(words)):
                if words[i] not in self.words:
                    raise Exception(
                        '%s not in dictionary, please use another word or set `soft` = True'
                        % (words[i])
                    )
        batches = np.array([self.get_vector_by_name(w) for w in words])
        return self._batch_process(
            batches,
            num_closest = num_closest,
            return_similarity = return_similarity,
        )

    def n_closest(
        self, word, num_closest = 5, metric = 'cosine', return_similarity = True
    ):
        """
        find nearest words based on a word.

        Parameters
        ----------
        word: str
            Eg, 'najib'
        num_closest: int, (default=5)
            number of words closest to the result.
        metric: str, (default='cosine')
            vector distance algorithm.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(word, str):
            raise ValueError('input must be a string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(metric, str):
            raise ValueError('metric must be a string')
        if not isinstance(return_similarity, bool):
            raise ValueError('num_closest must be a boolean')
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric = metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(
                self.get_vector_by_name(word).reshape((1, -1))
            )
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            wv = self.get_vector_by_name(word)
            closest_indices = self.closest_row_indices(
                wv, num_closest + 1, metric
            )
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            if word in word_list:
                word_list.remove(word)
            return word_list

    def closest_row_indices(self, wv, num, metric):
        dist_array = np.ravel(
            cdist(self._embed_matrix, wv.reshape((1, -1)), metric = metric)
        )
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    def analogy(self, a, b, c, num = 1, metric = 'cosine'):
        """
        analogy calculation, vb - va + vc.

        Parameters
        ----------
        a: str
        b: str
        c: str
        num: int, (default=1)
        metric: str, (default='cosine')
            vector distance algorithm.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(a, str):
            raise ValueError('a must be a string')
        if not isinstance(b, str):
            raise ValueError('b must be a string')
        if not isinstance(c, str):
            raise ValueError('c must be a string')
        if a not in self._dictionary:
            raise Exception('a not in dictinary')
        if b not in self._dictionary:
            raise Exception('b not in dictinary')
        if c not in self._dictionary:
            raise Exception('c not in dictinary')
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])
        return d_word_list

    def project_2d(self, start, end):
        """
        project word2vec into 2d dimension.

        Parameters
        ----------
        start: int
        end: int

        Returns
        -------
        tsne decomposition: numpy
        """
        if not isinstance(start, int):
            raise ValueError('start must be an integer')
        if not isinstance(end, int):
            raise ValueError('end must be an integer')
        tsne = TSNE(n_components = 2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list
