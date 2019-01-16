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
from . import home
from ._utils._utils import download_file
from ._models import _word2vec
from .texts._text_functions import simple_textcleaning


def load_wiki():
    """
    Return malaya pretrained wikipedia word2vec size 256

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
    Return malaya pretrained news word2vec

    Parameters
    ----------
    size: int, (default=256)

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    assert isinstance(size, int), 'input must be an integer'
    if size not in [32, 64, 128, 256, 512]:
        raise Exception(
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
    Train a word2vec for custom corpus

    Parameters
    ----------
    corpus: list
        list of strings
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    hidden_size: int, (default=256)
        vector size representation for hidden layer
    negative_samples_ratio: float, (default=0.5)
        negative samples ratio proportional to batch_size
    learning_rate: float, (default=0.01)
        learning rate for word2vec
    momentum: float, (default=0.9)
        momentum rate for optimizer=momentum
    epoch: int, (default=5)
        iteration numbers
    optimizer: str, (default='momentum')
        optimizer supported, ['gradientdescent', 'rmsprop', 'momentum', 'adagrad', 'adam']
    text_cleaning: function, (default=simple_textcleaning)
        function to clean the corpus

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    assert isinstance(corpus, str) or isinstance(
        corpus, list
    ), 'corpus must be a string or a list of string'
    assert vocab_size is None or isinstance(
        vocab_size, int
    ), 'vocab_size must be a None or an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(hidden_size, int), 'hidden_size must be an integer'
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(
        negative_samples_ratio, float
    ), 'negative_samples_ratio must be a float'
    assert isinstance(momentum, float), 'momentum must be a float'
    assert isinstance(embedding_noise, float), 'embedding_noise must be a float'
    assert isinstance(hidden_noise, float), 'hidden_noise must be a float'
    assert isinstance(learning_rate, float) or isinstance(
        learning_rate, int
    ), 'learning_rate must be a float or an integer'
    assert isinstance(optimizer, str), 'optimizer must be a string'
    assert batch_size > 0, 'batch_size must bigger than 0'
    assert epoch > 0, 'epoch must bigger than 0'
    assert embedding_size > 0, 'embedding_size must bigger than 0'
    assert hidden_size > 0, 'hidden_size must bigger than 0'
    assert (
        negative_samples_ratio > 0 and negative_samples_ratio <= 1
    ), 'negative_samples_ratio must bigger than 0 and less than or equal 1'
    assert (
        embedding_noise > 0 and embedding_noise <= 1
    ), 'embedding_noise must bigger than 0 and less than or equal 1'
    assert (
        hidden_noise > 0 and hidden_noise <= 1
    ), 'hidden_noise must bigger than 0 and less than or equal 1'
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

    def get_vector_by_name(self, word):
        """
        get vector based on name

        Parameters
        ----------
        word: str

        Returns
        -------
        vector: numpy
        """
        if word not in self._dictionary:
            arr = np.array([fuzz.ratio(word, k) for k in self.words])
            idx = (-arr).argsort()[:5]
            strings = ', '.join([self.words[i] for i in idx])
            raise Exception(
                'input not found in dictionary, here top-5 nearest words [%s]'
                % (strings)
            )
        return np.ravel(self._embed_matrix[self._dictionary[word], :])

    def tree_plot(
        self,
        labels,
        notebook_mode = False,
        figsize = (7, 7),
        annotate = True,
        figname = 'fig.png',
    ):
        """
        plot a tree plot based on output from calculator / n_closest / analogy

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig
        figsize : tuple, (default=(7, 7))
            figure size for plot
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
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
        plot a scatter plot based on output from calculator / n_closest / analogy

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy
        centre : str, (default=None)
            centre label, if a str, it will annotate in a red color
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig
        figsize : tuple, (default=(7, 7))
            figure size for plot
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        assert isinstance(labels, list), 'input must be a list'
        assert isinstance(
            notebook_mode, bool
        ), 'notebook_mode must be a boolean'
        assert isinstance(figsize, tuple), 'figsize must be a tuple'
        assert isinstance(figname, str), 'figname must be a string'
        assert isinstance(plus_minus, int), 'plus_minus must be an integer'
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        if centre:
            assert isinstance(centre, str), 'centre must be a string'
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

    def calculator(
        self,
        equation,
        num_closest = 5,
        metric = 'cosine',
        return_similarity = True,
    ):
        """
        calculator parser for word2vec

        Parameters
        ----------
        equation: str
            Eg, '(mahathir + najib) - rosmah'
        num_closest: int, (default=5)
            number of words closest to the result
        metric: str, (default='cosine')
            vector distance algorithm
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance

        Returns
        -------
        word_list: list of nearest words
        """
        assert isinstance(equation, str), 'input must be a string'
        assert isinstance(num_closest, int), 'num_closest must be an integer'
        assert isinstance(metric, str), 'metric must be a string'
        assert isinstance(
            return_similarity, bool
        ), 'num_closest must be a boolean'
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
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric = metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(
                _Calculator(tokens).exp().reshape((1, -1))
            )
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            closest_indices = self.closest_row_indices(
                _Calculator(tokens).exp(), num_closest + 1, metric
            )
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            return word_list

    def n_closest(
        self, word, num_closest = 5, metric = 'cosine', return_similarity = True
    ):
        """
        find nearest words based on a word

        Parameters
        ----------
        word: str
            Eg, 'najib'
        num_closest: int, (default=5)
            number of words closest to the result
        metric: str, (default='cosine')
            vector distance algorithm
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance

        Returns
        -------
        word_list: list of nearest words
        """
        assert isinstance(word, str), 'input must be a string'
        assert isinstance(num_closest, int), 'num_closest must be an integer'
        assert isinstance(metric, str), 'metric must be a string'
        assert isinstance(
            return_similarity, bool
        ), 'num_closest must be a boolean'
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
        analogy calculation, vb - va + vc

        Parameters
        ----------
        a: str
        b: str
        c: str
        num: int, (default=1)
        metric: str, (default='cosine')
            vector distance algorithm

        Returns
        -------
        word_list: list of nearest words
        """
        assert isinstance(a, str), 'a must be a string'
        assert isinstance(b, str), 'b must be a string'
        assert isinstance(c, str), 'c must be a string'
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
        project word2vec into 2d dimension

        Parameters
        ----------
        start: int
        end: int

        Returns
        -------
        tsne decomposition: numpy
        """
        assert isinstance(start, int), 'start must be an integer'
        assert isinstance(end, int), 'end must be an integer'
        tsne = TSNE(n_components = 2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list
