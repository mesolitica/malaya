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


def load(size = 256):
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
        raise Exception('size word2vec not supported')
    if not os.path.isfile('%s/word2vec-%d/word2vec.p' % (home, size)):
        print('downloading word2vec-%d embedded' % (size))
        download_file(
            'v7/word2vec/word2vec-%d.p' % (size),
            '%s/word2vec-%d/word2vec.p' % (home, size),
        )
    with open('%s/word2vec-%d/word2vec.p' % (home, size), 'rb') as fopen:
        return pickle.load(fopen)


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
